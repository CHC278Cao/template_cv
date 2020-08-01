# # encoding: utf-8
# """
# @author: ccj
# @contact:
# """
#
# import torch
# import torch.nn as nn
#
#
# class MultiheadAttentionClassifier(nn.Module):
#     def __init__(self, num_classes, ninp, nhead, dropout, attention_dropout=0.2):
#         super(MultiheadAttentionClassifier, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=ninp, num_heads=nhead, dropout=attention_dropout)
#         self.classifier = nn.Linear(ninp * 2, num_classes)
#         self.dropout = nn.Dropout(dropout)
#
#     def