from .utils import find_all_files, read_json, save_json

def main():
    files = find_all_files("summaries", ending="*.json")
    for f in files:
        data = read_json(f)
        segment = f.split('/')[-1].split('.')[0]
        data['segment'] = segment
        save_json(f, data)
        
if __name__ == "__main__":
    main()        
