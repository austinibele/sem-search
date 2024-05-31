from .utils import read_json, save_json
from .searcher import Searcher
from tqdm import tqdm

class Pipeline:
    def __init__(self, links_dict_path="links_dict.json", results_dir="results"):
        self.links_dict = read_json(links_dict_path) 
        self.results_dir = results_dir
        self.searcher = Searcher(dir="summaries")
        
    def run(self):
        for k, v in tqdm(self.links_dict.items()):
            print(f"Processing segment {k}...")
            
            article_results = {}
            for kk, vv in v.items():
                results = self.searcher.search(query=kk)
                article_results[kk] = results
            save_json(f"{self.results_dir}/{k}.json", article_results)
    
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
