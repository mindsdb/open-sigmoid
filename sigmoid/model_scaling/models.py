import copy

import torch

from sigmoid.local.preprocessing.coordinate_files import LocalLoader


class StochasticPool(torch.nn.Module):
    """ Implementation of a switch MoE.
    """

    def __init__(self, encoder, task):
        """ Initializes MoE.

        @param encoder: torch.nn.Module
            model used in soul-capturing step.
        @param task: str
            name of task: supported names are 'regression',
            'binary_classificatio' and 'classification
        """
        super(StochasticPool, self).__init__()

        self.enc_ = encoder
        for p in self.enc_.parameters():
            p.requires_grad = False

        self.ne_ = -1
        self.skills_ = torch.nn.ModuleList
        self.s_loss_ = None
        self.skill_metric_ = []
        self.skill_out_n_dim_ = -1

        self.switch_ = torch.nn.Module
        self.routes_ = torch.Tensor
        self.mask_ = torch.Tensor
        self.routing_ = torch.Tensor
        self.switch_loss_ = torch.nn.Module
        self.switch_metrics_ = {}
        self.balancing_ = torch.Tensor
        self.task_ = task

        self.temp_ = 4.0
        self.curr_skill_ = 0
        self.curr_skill_val_ = 0

        self.device_id_ = torch.device

    def set_device(self, device_id: torch.device):
        """ Set device to run training/inference

            @param device: torch.device
                instance of torch.device (ex: torch.device('cpu'))
        """
        self.device_id_ = device_id

    def set_skills(self, skill_model, n_skills, skill_loss):
        """ Set specialist skills.
        """
        self.ne_ = n_skills
        self.s_loss_ = skill_loss
        self.s_loss_.to(self.device_id_)

        self.skill_out_n_dim_ = skill_model.get_output_dims()[0]
        skill_model = skill_model.to(self.device_id_)
        self.skills_ = self._clone_module_list(skill_model, n_skills)

    def set_skill_metric(self, metric, **metric_kwargs):
        """ Sets metric to measure skill performance.

            @param metric: a not-yet-instanciated metric from torcheval.
        """
        self.skill_metric_ = []
        for _ in range(self.ne_):
            m = metric(**metric_kwargs)
            m.to(self.device_id_)
            self.skill_metric_.append(m)

    def set_switch(self, switch_model: torch.nn.Module):
        """ Sets switch.
        """
        self.switch_ = switch_model
        self.switch_.to(self.device_id_)

    def set_switch_balancing(self, class_weights):
        """ Sets routing balancing scheme.
        """
        w = torch.tensor(class_weights,
                         dtype=torch.float32,
                         device=self.device_id_)
        w = w / w.sum()
        self.balancing_ = w
        self.switch_loss_ = torch.nn.CrossEntropyLoss(weight=w)
        self.switch_loss_.to(self.device_id_)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """ Performs forward pass.

            The switching is performed on a per-batch basis, so that
            different "rows" in the batch get routed to different skills.
        """
        # outputs = torch.zeros((x_in.shape[0], self.skill_out_n_dim_),
        #                       dtype=torch.float32)
        # outputs = outputs.to(self.device_id_)

        _ = self.enc_(x_in)
        c = self.enc_.get_encodings()
        logits = self.switch_(c)
        #print("logits:", logits[torch.isnan(logits)])
        self.routing_ = torch.nn.functional.gumbel_softmax(logits, tau=self.temp_)
        self.routes_ = torch.argmax(self.routing_, dim=1)
        self.mask_ = self.routes_ == self.curr_skill_

        skill_output = self.skills_[self.curr_skill_](x_in[self.mask_, :])
        # outputs[self.mask_, :] = skill_output

        return skill_output, self.mask_

    def loss_fn(self, y_hat, y,
                alpha: float = 2.0,
                beta: float = 1.0,
                gamma: float = 1.0,
                eps: float = 1e-20) -> torch.Tensor:
        """ Computes loss of model and balancing loss of switch.
        """
        # balancing loss
        m = torch.sum(self.routing_)
        n = torch.sum(self.routing_[:, self.curr_skill_])
        x = n / (m + eps)
        b0 = 1.0 / self.ne_  # self.balancing_[self.curr_skill_]
        r = torch.pow(x - b0, alpha)
        p = torch.pow(x, beta)
        q = torch.pow(1.0 - x, gamma)
        b = r / (p * q + eps)

        # specialist loss
        s = 0.0
        if torch.sum(self.mask_) > 0:
            s = self.s_loss_(y_hat, y[self.mask_, :])

        return s + b

    def get_routes(self):
        """ Returns routes to experts.
        """
        return self.routes_

    def train_single_epoch(self,
                           data_loader: LocalLoader,
                           optimizer: torch.optim.Optimizer):
        """ Train auto-encoder for a single epoch.

            @param
        """
        self.train()
        epoch_loss = 0.0
        for x, y in data_loader:
            x = x.to(self.device_id_)
            y = y.to(self.device_id_)
            optimizer.zero_grad()
            y_hat, _ = self(x)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # update specialist to train
            self.curr_skill_ = self.curr_skill_ + 1
            self.curr_skill_ = self.curr_skill_ % int(self.ne_)

        return epoch_loss

    def eval_single_epoch(self, data_loader: LocalLoader):
        """ Runs model in validation data.
        """
        for metric in self.skill_metric_:
            metric.reset()

        self.eval()
        eval_loss = 0.0
        with torch.no_grad():
            updated = []
            for x, y in data_loader:

                self.curr_skill_ = self.curr_skill_val_

                x = x.to(self.device_id_)
                y = y.to(self.device_id_)
                y_hat, mask = self(x)
                loss = self.loss_fn(y_hat, y)
                eval_loss += loss.item()

                y_s = y[mask, :]
                e = self.curr_skill_val_
                if torch.sum(mask) > 0:
                    if self.task_ == 'classification':
                        pred = torch.argmax(y_hat, dim=1)
                        target = torch.argmax(y_s, dim=1)
                        self.skill_metric_[e].update(pred, target)
                    elif self.task_ == 'binary_classification':
                        pred = torch.zeros_like(y_hat,
                                                dtype=torch.int64,
                                                device=self.device_id_)
                        target = torch.zeros_like(y_s,
                                                dtype=torch.int64,
                                                device=self.device_id_)
                        target[y_s > 0.5] = 1
                        pred[y_hat > 0.5] = 1
                        self.skill_metric_[e].update(pred, target)
                    else:
                        self.skill_metric_[e].update(y_hat, y_s)
                    updated.append(e)

                # update specialist to train
                self.curr_skill_val_ = self.curr_skill_val_ + 1
                self.curr_skill_val_ = self.curr_skill_val_ % int(self.ne_)

            metric_vals = []
            for i, metric in enumerate(self.skill_metric_):
                if i not in updated:
                    metric_vals.append(-1.0)
                else:
                    metric_vals.append(metric.compute())

        return eval_loss, metric_vals

    def fit(self, train_loader, val_loader, n_epochs: int):
        """ Trains model on dataset.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1E-4)

        self.curr_skill_ = 0
        self.curr_skill_val_ = 0
        for epoch in range(n_epochs):
            epoch_loss = self.train_single_epoch(train_loader, optimizer)
            eval_loss, m_values = self.eval_single_epoch(val_loader)
            println = "{:+04d} {:4.4f} {:4.4f} {:4.4f}"
            for _ in range(len(self.skill_metric_)):
                println += " |{:+4.4f}| "
            print(println.format(epoch + 1, epoch_loss, eval_loss, self.temp_,
                                    *m_values))

            self.temp_ = self.temp_ * 0.99
            if self.temp_ < 0.1:
                self.temp_ = 0.1

    def evaluate(self, eval_loader) -> list:
        """ Runs model on evaluation dataset.
        """
        gt = []
        preds = []
        routes = []
        with torch.no_grad():
            for e in range(self.ne_):
                self.curr_skill_ = e
                for x, y in eval_loader:
                    x = x.to(self.device_id_)
                    y = y.to(self.device_id_)
                    y_hat, mask = self(x)
                    if torch.sum(mask) > 0:
                        r = self.get_routes()
                        preds.append(y_hat)
                        gt.append(y[mask, :])
                        routes.append(r[mask])

        predictions = torch.cat(preds).cpu().numpy()
        ground_truth = torch.cat(gt).cpu().numpy()
        skill_route = torch.cat(routes).cpu().numpy()

        return [ground_truth, predictions, skill_route, ]

    @staticmethod
    def _clone_module_list(module, n_clones: int):
        """
        ## Clone Module

        Make a `nn.ModuleList` with clones of a given module
        """
        return torch.nn.ModuleList(
            [copy.deepcopy(module) for _ in range(n_clones)])
