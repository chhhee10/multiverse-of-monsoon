"""Training entry point (the pipeline lives in training/).

  python -m training.build_assets                      # climatology, normalisation, static fields -> assets/
  python train.py --out runs/v1                        # train (GPU if available); see --help
  modal run training/modal_train.py --name v1          # same, on a Modal GPU
"""
from training.train import main

if __name__ == "__main__":
    main()
