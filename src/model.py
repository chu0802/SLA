import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models

from cdac_loss import BCE_softlabels, advbce_unlabeled, sigmoid_rampup
from evaluation import prediction


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


class ProtoClassifier(nn.Module):
    def __init__(self, size):
        super(ProtoClassifier, self).__init__()
        self.center = None
        self.label = None
        self.size = size

    def init(self, model, t_loader):
        t_pred, t_feat = prediction(t_loader, model)
        label = t_pred.argmax(dim=1)
        center = torch.nan_to_num(
            torch.vstack([t_feat[label == i].mean(dim=0) for i in range(self.size)])
        )
        invalid_idx = center.sum(dim=1) == 0
        if invalid_idx.any() and self.label is not None:
            old_center = torch.vstack(
                [t_feat[self.label == i].mean(dim=0) for i in range(self.size)]
            )
            center[invalid_idx] = old_center[invalid_idx]
        else:
            self.label = label
        self.center = center.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, T=1.0):
        dist = torch.cdist(x, self.center)
        return F.softmax(-dist * T, dim=1)


class ResBase(nn.Module):
    def __init__(self, backbone="resnet34", **kwargs):
        super(ResBase, self).__init__()
        self.res = models.__dict__[backbone](**kwargs)
        self.last_dim = self.res.fc.in_features
        self.res.fc = nn.Identity()

    def forward(self, x):
        return self.res(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_classes=65, temp=0.05):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.temp = temp

    def forward(self, x, reverse=False):
        x = self.get_features(x, reverse=reverse)
        return self.get_predictions(x)

    def get_features(self, x, reverse=False):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x)
        return F.normalize(x) / self.temp

    def get_predictions(self, x):
        return self.fc2(x)


class ResModel(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        hidden_dim=512,
        output_dim=65,
        temp=0.05,
        pre_trained=True,
    ):
        super(ResModel, self).__init__()
        self.f = ResBase(
            backbone=backbone,
            weights=models.__dict__[f"ResNet{backbone[6:]}_Weights"].DEFAULT
            if pre_trained
            else None,
        )
        self.c = Classifier(self.f.last_dim, hidden_dim, output_dim, temp)
        init_weights(self.c)

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.bce = BCE_softlabels()

    def forward(self, x, reverse=False):
        return self.c(self.f(x), reverse)

    def get_params(self, lr):
        params = []
        for k, v in dict(self.f.named_parameters()).items():
            if v.requires_grad:
                if "classifier" not in k:
                    params += [{"params": [v], "base_lr": lr * 0.1, "lr": lr * 0.1}]
                else:
                    params += [{"params": [v], "base_lr": lr, "lr": lr}]
        params += [{"params": self.c.parameters(), "base_lr": lr, "lr": lr}]
        return params

    def get_features(self, x, reverse=False):
        return self.c.get_features(self.f(x), reverse=reverse)

    def get_predictions(self, x):
        return self.c.get_predictions(x)

    def base_loss(self, x, y):
        return self.criterion(self.forward(x), y).mean()

    def feature_base_loss(self, f, y):
        return self.criterion(self.get_predictions(f), y).mean()

    def sla_loss(self, f, y1, y2, alpha):
        out = self.get_predictions(f)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def nl_loss(self, f, y, alpha, T):
        out = self.get_predictions(f)
        y2 = F.softmax(out.detach() * T, dim=1)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y)
        soft_loss = -(y2 * log_softmax_out).sum(dim=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def mme_loss(self, _, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        return lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))

    def cdac_loss(self, step, x, x1, x2):
        w_cons = 30 * sigmoid_rampup(step, 2000)
        f = self.f(x)
        f1 = self.f(x1)
        f2 = self.f(x2)

        out = self.c(f, reverse=True)
        out1 = self.c(f1, reverse=True)

        prob, prob1 = F.softmax(out, dim=1), F.softmax(out1, dim=1)
        aac_loss = advbce_unlabeled(
            target=None, f=f, prob=prob, prob1=prob1, bce=self.bce
        )

        out = self.c(f)
        out1 = self.c(f1)
        out2 = self.c(f2)

        prob, prob1, prob2 = (
            F.softmax(out, dim=1),
            F.softmax(out1, dim=1),
            F.softmax(out2, dim=1),
        )
        mp, pl = torch.max(prob.detach(), dim=1)
        mask = mp.ge(0.95).float()

        pl_loss = (F.cross_entropy(out2, pl, reduction="none") * mask).mean()
        con_loss = F.mse_loss(prob1, prob2)

        return aac_loss + pl_loss + w_cons * con_loss


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd
        return output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)
