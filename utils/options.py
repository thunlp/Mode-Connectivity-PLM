import argparse


class option:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Basic parameters
        parser.add_argument("--task_dir", default="data", required=True)
        parser.add_argument("--train_file", default="data", required=False)
        parser.add_argument("--dev_file", default="data", required=False)
        parser.add_argument("--test_file", default="data", required=False)
        parser.add_argument("--dataset", default="nlp_forest_single", required=False)
        parser.add_argument("--model", default="facebook/t5-base", required=False)
        parser.add_argument("--tokenizer_path", default="facebook/t5-base", required=False)

        parser.add_argument("--output_dir", default=None, type=str, required=True)
        parser.add_argument("--do_train", action='store_true')
        parser.add_argument("--do_valid", action='store_true')
        parser.add_argument("--do_predict", action='store_true')
        parser.add_argument("--predict_checkpoint", type=str, default="best-model.pt")

        # Model parameters
        parser.add_argument("--checkpoint", type=str)
        parser.add_argument("--do_lowercase", action='store_true', default=False)
        parser.add_argument("--freeze_embeds", action='store_true', default=False)

        # Preprocessing/decoding-related parameters
        parser.add_argument('--max_input_length', type=int, default=512)
        parser.add_argument('--max_output_length', type=int, default=64)
        parser.add_argument('--num_beams', type=int, default=4)
        parser.add_argument("--append_another_bos", action='store_true', default=False)

        # Training-related parameters
        parser.add_argument("--train_batch_size", default=32, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--predict_batch_size", default=32, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--warmup_proportion", default=0.01, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=0.01, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                            help="Max gradient norm.")
        parser.add_argument("--train_epochs", default=10000000, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_steps", default=500, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument("--warmup_rate", default=0.06)
        parser.add_argument("--lr_decay_style", default="constant")
        parser.add_argument("--train_iters", default=10000000, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument('--wait_step', type=int, default=10000000)

        # Other parameters
        parser.add_argument("--quiet", action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                            "A number of warnings are expected for a normal SQuAD evaluation.")
        parser.add_argument('--valid_interval', type=int, default=5000,
                            help="Evaluate & save model")
        parser.add_argument("--output_interval", type=int, default=10000)
        parser.add_argument("--log_interval", type=int, default=100)
        parser.add_argument("--early_stop", type=int, default=-1)
        parser.add_argument('--seed', type=int, default=20,
                            help="random seed for initialization")
        parser.add_argument("--choose_test_1000", action='store_true')
        parser.add_argument("--choose_dev_1000", action='store_true')
        parser.add_argument("--load_init_seed", type=int, default=20,
                            help="for loading the same initialization")

        # For tuning
        parser.add_argument("--learning_rate_list",
                            nargs="*", type=float, default=[])
        parser.add_argument("--bsz_list", nargs="*", type=int, default=[])
        parser.add_argument("--tune_method", type=str, help="model or adapter")
        parser.add_argument("--one_prefix", action='store_true')

        # For adapter
        parser.add_argument("--apply_adapter", action='store_true')
        parser.add_argument("--adapter_type", type=str, default='houlsby')
        parser.add_argument("--adapter_size", type=int, default=64)
        parser.add_argument("--r_mean", type=float, default=0)
        parser.add_argument("--r_std", type=float, default=0.02)

        # For interpolation
        parser.add_argument("--load_PET_path_1", type=str)
        parser.add_argument("--load_PET_path_2", type=str)
        parser.add_argument("--itpl_points", type=int, default=26)

        # For special settings
        parser.add_argument("--SGD_noise", action='store_true')
        parser.add_argument('--datasplit', type=str, default="")
        parser.add_argument('--cartography', action='store_true')
        parser.add_argument('--itp_on_train', action='store_true')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
