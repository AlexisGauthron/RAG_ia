import torch

def test_utilisation_GPU():
    print("\n")
    print("##################### TEST GPU #######################")
    if torch.cuda.is_available():
        print("\n")
        print("GPU is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print("\n")
        device = torch.device("cuda")
    else:
        print("\n")
        print("GPU is not available.")
        print("\n")
        device = torch.device("cpu")
    return device  

if __name__ == "__main__":
    test_utilisation_GPU()