import pandas as pd

# 读取 Excel 文件
df = pd.read_excel(r'D:\1Documents\gnnfakenews\code\HeterSGT-main\Data\gos\graph\edges\news2news.xlsx')

# 保存为 CSV 文件（使用逗号分隔，UTF-8 编码）
df.to_csv(r'D:\1Documents\gnnfakenews\code\HeterSGT-main\Data\gos\graph\edges\news2news.csv',
          index=False,  # 不保存行索引
          sep=',',      # 使用逗号作为分隔符
          encoding='utf-8')