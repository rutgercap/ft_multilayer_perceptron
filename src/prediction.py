from pathlib import Path
from sys import argv

from network import MLP

def main(): 
    # if len(argv) != 2:
    #     print("Usage: python prediction <model.json> <data.csv>")
    #     exit(1)
    # model_path = argv[1]
    # data_path = argv[2]

    model = MLP.from_file(Path("model.json"))
    print(model)


if __name__ == "__main__":
    main()
