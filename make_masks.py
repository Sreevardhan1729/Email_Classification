import pandas as pd
from utils import mask_pii
from tqdm import tqdm

df = pd.read_csv('data/raw_emails.csv')
out = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Masking emails"):
    masked, ents = mask_pii(row['email'])
    out.append({
        **row.to_dict(),
        'masked_body': masked,
        'entities': ents
    })

pd.DataFrame(out).to_csv('data/masked_emails.csv', index=False)
print("Masking complete. Output saved to data/masked_emails.csv")