import torch
import torch.nn as nn
import torch.nn.functional as F
from fontTools.misc.classifyTools import Classifier

def reciprocal_Loss(p, alpha, c):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    alp = E * (1 - label) + 1
    S1 = torch.sum(alp, dim=1, keepdim=True)
    reciprocal_Loss = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)+1/torch.sum(((1-label)*(
            torch.digamma(S1) -torch.digamma(alp))), dim=1, keepdim=True)
    return reciprocal_Loss
class EFN(nn.Module):
    def __init__(self,classes, dims, view):
        super(EFN, self).__init__()
        self.classes = classes
        self.view = view
        self.Classifiers = nn.ModuleList([Classifier(dims[i],self.classes) for i in range(self.view)])

    def infer(self,input):
        '''
        :param input: the data of aerial_image/ground_image
        :return: the evidence of both sense, a dict.
        '''
        evidence = dict()
        for view_num in range(self.view):
            evidence[view_num] = self.Classifiers[view_num](input[view_num])
        return evidence
    def Evidence_fusion(self,D_alpha):
        '''
        :param D_alpha: the Dirichlet distribution parameters of two sense
        :return: the Combined Dirichlet distribution parameters
        '''


        def Evidence_two(alpha1,alpha2):
            '''
            :param alpha1: Dirichlet distribution parameters of aerial image
            :param alpha2: Dirichlet distribution parameters of ground image
            :return: Combined Dirichlet distribution parameters
            '''
            alpha, E, c, S, u = dict(), dict(), dict(), dict(), dict()
            alpha[0] = alpha1
            alpha[1] = alpha2
            # calculate E,b,S,u of each view
            for view_num in range(len(alpha)):
                E[view_num] = alpha[view_num]-1
                S[view_num] = torch.sum(alpha[view_num], dim=1, keepdim=True)
                c[view_num] = E[view_num] / (S[view_num].expand(E[view_num].shape))
                u[view_num] = self.classes / S[view_num]
            # c0 @ c1
            cc = torch.bmm(c[0].view(-1, self.classes, 1), c[1].view(-1, 1, self.classes))
            u1_expand = u[0].expand(c[0].shape)
            # c0 * (1-u0)
            view1_weight = torch.mul(c[0],(1-u1_expand))
            u2_expand = u[1].expand(c[0].shape)
            # c1 * (1-u1)
            view2_weight = torch.mul(c[1],(1-u2_expand))
            # c0 * (1-u0) of one view all classes
            v1weigt_all = torch.mul((1-u1_expand),(1-u1_expand))
            # c1 * (1-u1) of one view all classes
            v2weigt_all = torch.mul((1-u2_expand),(1-u2_expand))


            cc_diag = torch.diagonal(cc,dim1=-2,dim2=-1).sum(-1)
            # calculate b_after
            c_total = (torch.mul(c[0], c[1]) + view1_weight + view2_weight) /((cc_diag.view(-1, 1).expand(c[0].shape))
                                                                          +v1weigt_all+v2weigt_all+torch.mul((1-u1_expand),(1-u2_expand)))
            # calculate u_after
            u_total = torch.mul(1-u[0], 1-u[1]) / ((cc_diag.view(-1, 1).expand(u[0].shape))+torch.mul(1-u[0],1-u[0])
                                           +torch.mul(1-u[1],1-u[1])+torch.mul(1-u[0], 1-u[1]))

            # calculate S_after
            S_total = self.classes / u_total

            # calculate E_after
            E_total = torch.mul(c_total, S_total.expand(c_total.shape))
            # calculate alpha_after
            alpha_total = E_total + 1
            return alpha_total

        alpha_after = Evidence_two(D_alpha[0], D_alpha[1])
        return alpha_after

    def forward(self,X,y):
        evidence = self.infer(X)
        loss = 0
        D_alpha = dict()
        for view_num in range(len(X)):
            D_alpha[view_num] = evidence[view_num] + 1
            loss += reciprocal_Loss(y,D_alpha[view_num],self.classes)

        alpha_after = self.Evidence_fusion(D_alpha)
        evidence_after = alpha_after - 1
        loss += reciprocal_Loss(y,D_alpha[view_num],self.classes)
        loss = torch.mean(loss)
        return evidence,evidence_after,loss



class Classifier(nn.Module):
    '''
    Feature extraction
    '''
    def __init__(self, dims, classes):
        super(Classifier,self).__init__()
        self.classes = classes
        self.fc = nn.ModuleList()
        self.num_layers = len(dims)
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(dims[i], dims[i + 1]))
        self.fc.append(nn.Linear(dims[self.num_layers-1], 2048))
        self.fc.append(nn.ReLU(True))
        self.fc.append(nn.BatchNorm1d(2048))
        self.fc.append(nn.Dropout())
        self.fc.append(nn.Linear(2048, self.classes))
        self.fc.append(nn.Softplus())
    def forward(self,input):
        '''
        :param input: the input data of one sense
        :return: the evidence of one sense
        '''
        feature = self.fc[0](input)
        for i in range(1,len(self.fc)):
            feature = self.fc[i](feature)
        return feature