import torch
from transformers import DistilBertModel
from transformers.utils.fx import symbolic_trace

def generate_dbert_mlir(traced_model: torch.fx.GraphModule) -> str:
    mlir_lines = [
        "module {",
        "  func.func @forward(%input_ids: tensor<1x128xi32>) -> tensor<1x128x768xf32> {"
    ]
    
    node_to_ssa = {}
    ssa_counter = 0
    
    for node in traced_model.graph.nodes:
        # 1. Handle Inputs
        if node.op == "placeholder":
            node_to_ssa[node.name] = "%input_ids"
            continue
            
        # 2. Handle Outputs
        if node.op == "output":
            # HF outputs are usually tuples
            ret_node = node.args[0][0] if isinstance(node.args[0], tuple) else node.args[0]
            ret_val = node_to_ssa.get(ret_node.name if hasattr(ret_node, 'name') else "", "%unknown")
            mlir_lines.append(f"    func.return {ret_val} : tensor<1x128x768xf32>")
            break
            
        # Generate new SSA register for this operation's output
        ssa_val = f"%{ssa_counter}"
        ssa_counter += 1
        node_to_ssa[node.name] = ssa_val
        
        # Collect SSA operands
        operands = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                operands.append(node_to_ssa.get(arg.name, "%unknown"))
        
        op_str = ", ".join(operands)
        target = str(node.target).lower()

        # 3. Pattern Match PyTorch Ops to dbert MLIR Ops
        if "matmul" in target or "bmm" in target or "linear" in target:
            # We assume tensor<*xf32> for simplicity in this frontend
            mlir_lines.append(f"    {ssa_val} = \"dbert.matmul\"({op_str}) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>")
        
        elif "softmax" in target:
            mlir_lines.append(f"    {ssa_val} = \"dbert.softmax\"({op_str}) : (tensor<*xf32>) -> tensor<*xf32>")
        
        elif "layer_norm" in target or "layernorm" in target:
            mlir_lines.append(f"    {ssa_val} = \"dbert.layernorm\"({op_str}) : (tensor<*xf32>) -> tensor<*xf32>")
        
        elif "gelu" in target:
            mlir_lines.append(f"    {ssa_val} = \"dbert.gelu\"({op_str}) : (tensor<*xf32>) -> tensor<*xf32>")
            
        elif "add" in target:
            mlir_lines.append(f"    {ssa_val} = \"dbert.add\"({op_str}) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>")
            
        elif "embedding" in target:
            mlir_lines.append(f"    {ssa_val} = \"dbert.embeddings\"({op_str}) : (tensor<*xi32>) -> tensor<*xf32>")
            
        else:
            # Fallback for unrecognized PyTorch nodes (e.g., getattr, view, reshape)
            mlir_lines.append(f"    // Ignored or generic op: {node.op} target={target}")
            # We map it to a generic operation just to keep the SSA chain alive
            mlir_lines.append(f"    {ssa_val} = \"dbert.generic\"({op_str}) {{target=\"{target}\"}} : () -> tensor<*xf32>")
                
    mlir_lines.extend([
        "  }",
        "}"
    ])
    return "\n".join(mlir_lines)

def export_to_mlir(model_name="distilbert-base-uncased", output_file="distilbert.mlir"):
    print(f"Loading {model_name} from HuggingFace...")
    model = DistilBertModel.from_pretrained(model_name)
    
    print("Tracing model execution graph with torch.fx...")
    input_names = ["input_ids", "attention_mask"]
    
    # HuggingFace provides a custom tracer to handle their complex architectures
    traced_model = symbolic_trace(model, input_names=input_names)
    
    print("Dynamically mapping PyTorch FX nodes to MLIR dbert ops...")
    mlir_code = generate_dbert_mlir(traced_model)
    
    with open(output_file, "w") as f:
        f.write(mlir_code)
    
    print(f"Successfully exported PyTorch graph to: {output_file}")
    print(f"Generated {len(traced_model.graph.nodes)} MLIR operations.")
    print("You can now compile it: `distilbert-opt distilbert.mlir --dbert-fuse-attention`")

if __name__ == "__main__":
    export_to_mlir()
