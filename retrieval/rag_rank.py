import pandas as pd
import torch
import argparse
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


class ProfileRAGSorter:
    def __init__(self, model_name, device=None, print_examples=1):
        self.print_limit = print_examples
        self.print_count = 0
        
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
        print(f"正在加载模型: {model_name} (使用设备: {self.device})...", flush=True)
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            print(f"模型加载失败: {e}", flush=True)
            raise

    def sort_batch(self, df_batch, profile_col='profile'):
       
        all_queries = []
        corpus_map = []  
        original_data_map = [] 

        for _, row in df_batch.iterrows():
            profile_data = row.get(profile_col, None)
            
            if not isinstance(profile_data, np.ndarray):
                profile_data = np.array([]) if profile_data is None else np.array(profile_data)

            def safe_str(val): return str(val) if val is not None else ""
            q_main = safe_str(row.get('question'))
            t_main = safe_str(row.get('question_text'))
            query_text = f"{q_main} {t_main}".strip()
            all_queries.append(query_text)

            corpus_texts = []
            valid_indices = []
            if profile_data.size > 0:
                for idx, item in enumerate(profile_data):
                    if isinstance(item, dict):
                        q_item = safe_str(item.get('question'))
                        t_item = safe_str(item.get('question_text'))
                        combined_text = f"{q_item}\n{t_item}".strip()
                        if combined_text:
                            corpus_texts.append(combined_text)
                            valid_indices.append(idx)
            
            corpus_map.append(corpus_texts)
            original_data_map.append({
                "original_array": profile_data,
                "valid_indices": valid_indices,
                "has_work": bool(query_text and corpus_texts)
            })

        corpus_counts = [len(c) for c in corpus_map]
        all_corpus_texts = [text for sublist in corpus_map for text in sublist]

        if not all_corpus_texts:
            return [d['original_array'] for d in original_data_map]

        with torch.no_grad():
            query_embeddings = self.model.encode(all_queries, convert_to_tensor=True, show_progress_bar=False)
            corpus_embeddings = self.model.encode(all_corpus_texts, convert_to_tensor=True, show_progress_bar=False)

        sorted_results = []
        corpus_start_idx = 0
        for i in range(len(df_batch)):
            row_data = original_data_map[i]
            
            if not row_data["has_work"]:
                sorted_results.append(row_data["original_array"])
                continue

            num_corpus_items = corpus_counts[i]
            corpus_end_idx = corpus_start_idx + num_corpus_items

            q_emb = query_embeddings[i]
            c_embs = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            cos_scores = util.cos_sim(q_emb, c_embs)[0]
            scores_np = cos_scores.cpu().numpy()
            
            relative_sort_idx = np.argsort(-scores_np)
            
            valid_indices = row_data["valid_indices"]
            final_indices = [valid_indices[j] for j in relative_sort_idx]
            
            sorted_profile_array = row_data["original_array"][np.array(final_indices)]
            sorted_results.append(sorted_profile_array)

            corpus_start_idx = corpus_end_idx

            if self.print_count < self.print_limit:
                print(f"\n{'='*20} Processing Example {self.print_count + 1} (from Batch) {'='*20}", flush=True)
                print(f"Query: {all_queries[i][:60]}...")
                print(f"Top 1 Score: {scores_np[relative_sort_idx[0]]:.4f}", flush=True)
                self.print_count += 1
                
        return sorted_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--profile_col', type=str, default='profile')
    parser.add_argument('--save_original', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of rows to process in a single batch.')
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"错误: 文件不存在 -> {args.input_file}")
        return

    print(f"正在读取数据: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    print(f"数据加载完成，共 {len(df)} 行。")

    sorter = ProfileRAGSorter(model_name=args.model_name, device=args.device)

    print(f"开始进行 RAG 排序 (批处理大小: {args.batch_size})...")
    
    if args.save_original:
        print(f"保留原始列: {args.profile_col}_original")
        df[f'{args.profile_col}_original'] = df[args.profile_col]
    
    
    num_batches = max(1, len(df) // args.batch_size)
    batches = np.array_split(df, num_batches)
    
    all_sorted_profiles = []
    for batch_df in tqdm(batches, desc="正在处理批次"):
        sorted_batch = sorter.sort_batch(batch_df, profile_col=args.profile_col)
        all_sorted_profiles.extend(sorted_batch)
    
    df[args.profile_col] = all_sorted_profiles

    print(f"保存结果到: {args.output_file}")
    df.to_parquet(args.output_file, index=False)
    print("完成！")

if __name__ == "__main__":
    main()