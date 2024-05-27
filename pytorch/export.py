import torch
import sys
from diffusers.models.autoencoder_tiny import AutoencoderTiny
#from diffusers.models.vae import DecoderOutput
from torch.utils._pytree import PyTree


class Exporter:
    def __init__(self, model_cls):
        self.model_cls = model_cls
                                                                                                                                                                                                            
    def export(self, output):                                                                                                                                                                               
        model = self.model_cls()
        model.eval()  # Set the model to evaluation mode                                                                                                                                                    

        # Create a dummy input                                                                                                                                                                              
        dummy_input = torch.randn(3, 32, 32, dtype=torch.float32)  # Example input shape (batch_size, sequence_length)
                                                                                                                                                                                                            
        # Export the model to a graph module                                                                                                                                                                
        graph_module = torch.export.export(model, (dummy_input,))                                                                                                                                           
                                                                                                                                                                                                            
        # Save the graph module
        torch.export.save(graph_module, output)
        torch.export.load(output)

def export_autoencoder():
    e = Exporter(AutoencoderTiny)
    with torch.no_grad():
        e.export('autoencoder_model.pt2')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No argument provided.")
        sys.exit(1)

    function_name = f"export_{sys.argv[1]}"
    if function_name in globals():
        globals()[function_name]()
    else:
        print("Invalid argument.")
