import os
import json

dir = r'D:\cours\MA2\Projet_ML\datas\all-rnr-annotated-threads\charliehebdo-all-rnr-threads'
def read_data(dir):
    data = []
    # non-rumour
    non_rumours_path = os.path.join(dir, 'non-rumours')

    for threads in os.listdir(non_rumours_path):
        # pour check que les folders
        if threads.startswith('.') or not os.path.isdir(os.path.join(non_rumours_path, threads)):
            continue
        #threads= folder of 1 thread: 2 subfolders 'reactions' and 'source-tweets'
        thread_path = os.path.join(non_rumours_path, threads)
        source_path = os.path.join(thread_path, 'source-tweets')

        if os.path.exists(source_path):  # Check if source-tweet directory exists
            for file in os.listdir(source_path):
                if file.endswith('.json'):
                    file_path = os.path.join(source_path, file)
                    print(file_path)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Skip files with macOS attribute metadata
                            if "ATTR" in content[:50] or "Apple" in content[:50]:
                                print(f"Skipping macOS attribute file: {file_path}")
                                continue
                            
                            try:
                                json_data = json.loads(content)
                                print(json.dumps(json_data, indent=2)[:500])
                                data.append(json_data)
                                exit()
                            except json.JSONDecodeError:
                                continue
                    except UnicodeDecodeError:
                        try:
                            # Try with latin-1 encoding as fallback
                            with open(file_path, 'r', encoding='latin-1') as f:
                                content = f.read()
                                # Skip files with macOS attribute metadata
                                if "ATTR" in content[:50] or "Apple" in content[:50]:
                                    print(f"Skipping macOS attribute file: {file_path}")
                                    continue
                                
                                # Try to parse as JSON to verify format
                                try:
                                    json_data = json.loads(content)
                                    print(json.dumps(json_data, indent=2)[:500])  # Print first 500 chars only
                                    data.append(json_data)
                                    exit()
                                except json.JSONDecodeError:
                                    print(f"Invalid JSON format in file: {file_path}")
                                    continue
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
                        
        reactions_path = os.path.join(thread_path, 'reactions')

    return data

read_data(dir)