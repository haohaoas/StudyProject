import torch
import torch.nn.functional as F

# 假设老师和学生的输出 logits（batch=1, num_classes=3）
teacher_logits = torch.tensor([[2.0, 0.5, 0.3]])
student_logits = torch.tensor([[1.5, 0.7, 0.8]])
labels=torch.tensor([0])
# 设置温度
T = 2.0

# 老师的概率分布
teacher_probs = F.softmax(teacher_logits / T, dim=1)         # 1 x 3
print("teacher_probs:", teacher_probs)

# 学生的 log 概率分布
student_log_probs = F.log_softmax(student_logits / T, dim=1) # 1 x 3
print("student_log_probs:", student_log_probs)

# 计算KL散度
kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T ** 2)

ce_loss=F.cross_entropy(student_logits, labels)

alpha=0.7
total_loss=alpha*ce_loss+(1-alpha)*kl_loss
print("CE损失：", ce_loss)

print("KL蒸馏损失：", kl_loss)