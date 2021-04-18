from config import Hyper, OutputStore
from train import train
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# It all starts here
def main():
    print("\n"*10)
    print("-"*100)
    print("Start of Person Detection in Images")
    OutputStore.set_output_stores()     # Ensures the folders exist for output
    Hyper.display()
    train()
    print("-"*100)
    print("\n"*5)
    print("-"*100)
    Hyper.display()
    print("End of Person Detection in Images")
    print("-"*100)


if __name__ == "__main__":
    main()