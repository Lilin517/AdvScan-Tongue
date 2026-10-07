import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):

    def __init__(self, n_prototypes: int, teacher_temp: float=0.04, student_temp: float=0.1, center_momentum: float=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, n_prototypes))

    def forward(self, teacher_logits_list, student_logits_list):
        teacher_outs = []
        for t_out in teacher_logits_list:
            t_out = t_out.detach()
            t_out = t_out - self.center
            t_out = F.softmax(t_out / self.teacher_temp, dim=-1)
            teacher_outs.append(t_out)
        teacher_out = torch.stack(teacher_outs).mean(dim=0)
        loss = 0.0
        n_student = len(student_logits_list)
        for s_out in student_logits_list:
            s_out = F.log_softmax(s_out / self.student_temp, dim=-1)
            loss += -(teacher_out * s_out).sum(dim=-1).mean()
        loss = loss / n_student
        self._update_center(teacher_logits_list)
        return loss

    @torch.no_grad()
    def _update_center(self, teacher_logits_list):
        batch_center = torch.cat(teacher_logits_list).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
