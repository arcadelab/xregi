import json
import os
  
# Read Existing JSON File
def config_json(xray_path, net_path):
    with open('config.json') as f:
        data = json.load(f)
    f.close()
    
    ## add args into config.json
    current_path = os.path.abspath(os.path.dirname(__file__))
    data['xray_path'] = os.path.join(current_path, xray_path)
    data['nets'] = os.path.join(current_path, net_path)
    data['label_path'] = os.path.join(current_path, "data/real_label.h5")
    data['output_path'] = os.path.join(current_path, "data")
    data['input_data_file_path'] = os.path.join(current_path, "data/synthex_input.h5")
    data['input_label_file_path'] = os.path.join(current_path, "data/synthex_label_input.h5")
    data['output_data_file_path'] = os.path.join(current_path, "data/output.h5")
    data['heat_file_path'] = os.path.join(current_path, "data/output.h5")
    data['out'] = os.path.join(current_path, "data/own_data.csv")

        
    # Create new JSON file
    with open('config.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    # Closing file
    f.close()
if __name__ == "__main__":
    config_json("data/xray", "data/yy_checkpoint_net_20.pt")

