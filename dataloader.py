from torch.utils.data import Dataset
from utils import *

import os

from tdc.generation import MolGen
from tdc.single_pred import ADME
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
import pandas as pd
import numpy as np
from collections import Counter

from transformers import AutoTokenizer, AutoModel, T5EncoderModel
import torch
from tqdm import tqdm

def get_splits(data):
    split = data.get_split()
    split = [split['train'], split['valid'], split['test']]
    split = pd.concat(split)
    return split

def get_drug_embeddings(smiles_list, tokenizer, model, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size)):
        batch = smiles_list[i:i+batch_size]
        encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding["input_ids"].to('cuda')
        attention_mask = encoding["attention_mask"].to('cuda')  # Get the attention mask
        
        with torch.no_grad():
            # Pass both input_ids and attention_mask to the model
            output = model(input_ids, attention_mask=attention_mask)
            # You can use either the last hidden state or the pooled output
            # depending on your requirements. Here we're using the mean of the last hidden state.
            embedding = output.last_hidden_state.mean(1)
            embeddings.extend(embedding.cpu().numpy())
            
    return embeddings

class PKDataset(Dataset):
    def __init__(self, path, embeddings_path) -> None:
        super().__init__()

        f = pd.read_csv(path)
        drug_embeddings = np.load(embeddings_path)
        smiles = f['Drug'].values
        vlists = {
            col: f[col].values for col in f.drop(labels=['Drug'], axis=1).columns 
        }
        inmask = remove_outliers([v for _,v in vlists.items()])
        print(sum(inmask))
        smiles = smiles[inmask]
        vlists = {
            k: v[inmask] for k,v in vlists.items()
        }

        nullmask = np.stack([
            np.isnan(v)==False for _,v in vlists.items()
            ], axis=-1)

        vlists = {
            k: norm(v) for k,v in vlists.items()
        }

        self.dmss = []
        for k,v in vlists.items():
            vlists[k], dms = sample_local_gaussian(v)
            self.dmss.append(dms)

        # TODO: Train models here to infill values better!

        self.dataset = []
        for i, gt in enumerate(zip(*[v for _,v in vlists.items()])):
            self.dataset.append({
                "sm": smiles[i],
                "ft": drug_embeddings[i],
                "ma": nullmask[i],
                "gt": np.array(gt)
            })
        print(len(self.dataset))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def update(self, idx, delta):
        item = self.dataset[idx]["gt"]
        self.dataset[idx]["gt"] = item + delta

class PKDataloader:
    def __init__(
        self,
        embed_model_name,
        data_dir,
    ):
        self.embed_model = {
            "t5":'sagawa/PubChem-10m-t5-v2',
            "deberta":'sagawa/PubChem-10m-deberta',
            'chemberta_zinc': 'seyonec/ChemBERTa-zinc-base-v1',
            'chemberta_10m': 'DeepChem/ChemBERTa-10M-MLM',
            'MHGGNN': None

        }[embed_model_name]

        self.data_dir = data_dir
        self.data_dfs = {}

        self.save_data_file_name = os.path.join(data_dir, "xtended_data_all.csv")
        self.drug_embed_file_name = os.path.join(data_dir, f"xtended_emb_all_{embed_model_name}.npy")

        if not os.path.exists(self.save_data_file_name):
            self._download()
            self._build_dataset()

        if not os.path.exists(self.drug_embed_file_name):
            assert(embed_model_name != 'MHGGNN')
            self._generate_embeddings()

        self.dataset = PKDataset(self.save_data_file_name, self.drug_embed_file_name)

    def _download(
        self
    ):
        adme_names = [
            'Caco2_Wang',
            'PPBR_AZ',
            'VDss_Lombardo',
            'Half_Life_Obach'
        ]

        for name in adme_names:
            data = ADME(name = name, path=self.data_dir)
            self.data_dfs[name] = get_splits(data)

        data = Tox(name = 'LD50_Zhu', path=self.data_dir)
        self.data_dfs['LD50_Zhu'] = get_splits(data)


    def _build_dataset(
        self
    ):
        # Populate smiles
        smiles = []
        for _, df in self.data_dfs.items():
            smiles.extend(list(df['Drug']))
        smiles = list(set(smiles))
        print(len(smiles))

        # Build dataset
        dataset = {'Drug': smiles}
        for k, df in self.data_dfs.items():
            df = {v['Drug']: v['Y'] for v in df.to_dict(orient='records')}
            dataset[k] = []
            for smile in smiles:
                dataset[k].append(df.get(smile, np.nan))
            print(len(dataset[k])-sum(np.isnan(np.array(dataset[k]))))

        print(dataset.keys())

        self.df = pd.DataFrame(data=dataset)

        self.df.to_csv(self.save_data_file_name, index=False)


    def _generate_embeddings(
        self
    ):
        drug_tokenizer = AutoTokenizer.from_pretrained(self.embed_model)
        drug_model = T5EncoderModel.from_pretrained(self.embed_model)

        drug_model.to('cuda')

        self.drug_embeddings = get_drug_embeddings(list(self.df['Drug']), drug_tokenizer, drug_model)

        np.save(self.drug_embed_file_name, np.array(self.drug_embeddings))
        
        
        