import os
import torch
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """Base class for CycleGAN-style models."""
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = Path(opt.checkpoints_dir) / opt.name  # save all the checkpoints to save_dir
        self.device = opt.device
        # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        if opt.preprocess != "scale_width":
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net = networks.init_net(net, opt.init_type, opt.init_gain)

                if not self.isTrain or opt.continue_train:
                    load_suffix = f"iter_{opt.load_iter}" if opt.load_iter > 0 else opt.epoch
                    load_filename = f"{load_suffix}_net_{name}.pth"
                    load_path = self.save_dir / load_filename

                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        net = net.module
                    print(f"loading the model from {load_path}")

                    state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)

                    if hasattr(state_dict, "_metadata"):
                        del state_dict._metadata

                    for key in list(state_dict.keys()):
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                    net.load_state_dict(state_dict)

                net.to(self.device)

                if dist.is_initialized():
                    if self.opt.norm == "syncbatch":
                        raise ValueError(f"For distributed training, opt.norm must be 'syncbatch' or 'inst', but got '{self.opt.norm}'. " "Please set --norm syncbatch for multi-GPU training.")

                    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.device.index])
                    dist.barrier()

                setattr(self, "net" + name, net)

        self.print_networks(opt.verbose)

        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"learning rate {old_lr:.7f} -> {lr:.7f}")

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, "loss_" + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        if not dist.is_initialized() or dist.get_rank() == 0:
            for name in self.model_names:
                if isinstance(name, str):
                    save_filename = f"{epoch}_net_{name}.pth"
                    save_path = self.save_dir / save_filename
                    net = getattr(self, "net" + name)

                    if hasattr(net, "module"):
                        model_to_save = net.module
                    else:
                        model_to_save = net

                    if hasattr(model_to_save, "_orig_mod"):
                        model_to_save = model_to_save._orig_mod

                    torch.save(model_to_save.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys): 
            if module.__class__.__name__.startswith("InstanceNorm") and (key == "running_mean" or key == "running_var"):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (key == "num_batches_tracked"):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{epoch}_net_{name}.pth"
                load_path = self.save_dir / load_filename
                net = getattr(self, "net" + name)

                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                print(f"loading the model from {load_path}")

                state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)

                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

        if dist.is_initialized():
            dist.barrier()

    def print_networks(self, verbose):
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f"[Network {name}] Total number of parameters : {num_params / 1e6:.3f} M")
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def init_networks(self, init_type="normal", init_gain=0.02):
        import os

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)

                if torch.cuda.is_available():
                    if "LOCAL_RANK" in os.environ:
                        local_rank = int(os.environ["LOCAL_RANK"])
                        net.to(local_rank)
                        print(f"Initialized network {name} with device cuda:{local_rank}")
                    else:
                        net.to(0)
                        print(f"Initialized network {name} with device cuda:0")
                else:
                    net.to("cpu")
                    print(f"Initialized network {name} with device cpu")

                networks.init_weights(net, init_type, init_gain)
