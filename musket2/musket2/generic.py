import os
import sys
import torch
from musket2 import configloader,context,utils,datasets,callbacks,losses,binding_platform,models,metrics
from musket2.utils import ensure,key_or_default
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import DiskSaver,Checkpoint
from ignite.metrics import Accuracy, Loss,TopKCategoricalAccuracy
import numpy as np
import importlib


def model_function(func):
    func.model=True
    return func

def _shape(x):
    if isinstance(x,tuple):
        return [_shape(i) for i in x]
    if isinstance(x,list):
        return [_shape(i) for i in x]
    if hasattr(x, 'shape'):
        return x.shape
    return None


class ExecutionConfig:

    def __init__(self, fold=0, stage=0, subsample=1.0, dr: str = "", drawingFunction=None):
        self.subsample = subsample
        self.stage = stage
        self.fold = fold
        self.dirName = dr
        self.drawingFunction = drawingFunction
        pass

    def predictions_dump(self, y=False):
        ensure(os.path.join(self.dirName, "predictions"))
        if y:
            return os.path.join(self.dirName, "predictions",
                                "validation" + str(self.fold) + "." + str(self.stage) + "-gt.csv")
        else:
            return os.path.join(self.dirName, "predictions",
                                "validation" + str(self.fold) + "." + str(self.stage) + "-pt.csv")

    def predictions_holdout(self, y=False):
        ensure(os.path.join(self.dirName, "predictions"))
        if y:
            return os.path.join(self.dirName, "predictions",
                                "holdout-" + str(self.fold) + "." + str(self.stage) + "-gt.csv")
        else:
            return os.path.join(self.dirName, "predictions",
                                "holdout-" + str(self.fold) + "." + str(self.stage) + "-pt.csv")

    def weightsPath(self):
        ensure(os.path.join(self.dirName, "weights"))
        return os.path.join(self.dirName, "weights", "best-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def last_weightsPath(self):
        ensure(os.path.join(self.dirName, "weights"))
        return os.path.join(self.dirName, "weights", "last-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def classifier_weightsPath(self):
        ensure(os.path.join(self.dirName, "classify_weights"))
        return os.path.join(self.dirName, "classify_weights",
                            "best-" + str(self.fold) + "." + str(self.stage) + ".weights")

    def metricsPath(self):
        ensure(os.path.join(self.dirName, "metrics"))
        return os.path.join(self.dirName, "metrics", "metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")

    def classifier_metricsPath(self):
        ensure(os.path.join(self.dirName, "classify_metrics"))
        return os.path.join(self.dirName, "classify_metrics",
                            "metrics-" + str(self.fold) + "." + str(self.stage) + ".csv")

class GenericConfig:

    def __init__(self,**cfg):
        self.folds_count=5
        self.stratified=False
        self.rs=123
        self.primary_metric="val_loss"
        self.primary_metric_mode = "min"
        self.validation_split=0.2
        self.test_split=0
        self.drop_last=True
        self.aug=None
        self.num_workers=0
        self.imports=[]
        self.dataset=None
        self.test=None
        self.optimizer="adam"
        self.schedule = key_or_default("schedule",cfg,None)
        self.loss="binary_crossentropy"
        self.early_stopping=key_or_default("early_stopping",cfg,None)
        def batchPrepare(batch, device, non_blocking):
            return batch["x"].cuda(), batch["y"].cuda()
        self.batchPrepare=batchPrepare

        def outputTransform(y, y1, y2):
            return y1, (y2 > 0.5).type_as(y1)

        self.outputTransform = outputTransform
        for v in cfg:
             val = cfg[v]
             setattr(self, v, val)
        self.cfg_dict=cfg
        self.stages=[self.create_stage(c,cfg["stages"][c]) for c in range(len(cfg["stages"]))]
        if isinstance(self.metrics,str):
            self.metrics=[self.metrics]
        if isinstance(self.final_metrics, str):
            self.final_metrics = [self.final_metrics]
        pass

    def create_stage(self,num,stage_dict):
        return TrainingStage(self,num,stage_dict)

    def create_model(self)->torch.nn.Module:
        return binding_platform.create_extension(models.module,self.cfg_dict["architecture"],None,{})

    def get_dataset(self,name)->datasets.DataSet:
        raise ValueError()

    def get_split(self,ds,fold:int=0)->datasets.DataSplit:
        test=None
        return datasets.KFoldedSplitter(ds, self.folds_count, self.stratified,self.rs, self.validation_split,self.test_split,
                 self.test)[fold]

    def create_loader(self,ds):
        raise ValueError()


    def fit(self):
        model=self.create_model()
        for st in self.stages:
            st.fit(model)
        self.create_summary()
        pass

    def create_summary(self):
        results={}
        toCalc=[]
        for m in self.final_metrics:
            if binding_platform.extension(metrics.final_metric,m):
                results[m]=binding_platform.create_extension(metrics.final_metric,m,self)
            else:
                toCalc.append(m)
        if len(toCalc)>0:
            for m in toCalc:
                mValues=[]
                for i in range(self.folds_count):
                    ds="validation"
                    if self.test_split>0:
                        ds="holdout"
                    preds=self.predictions(ds,i)

                    mValues.append(binding_platform.create_extension(metrics.metric, m, preds))
                mean=np.mean(mValues)
                maxV=np.max(mValues)
                minV=np.min(mValues)
                results[m]=mean
                results[m+"_max"]=maxV
                results[m+"_min"]=minV

        results["result"]=results[self.experiment_result]
        utils.save_yaml(os.path.join(os.path.dirname(self.path),"summary.yaml"),results)

    def load_model(self,fold=0,stage=-1):
        return self.stages[stage].load_model(fold)

    def create_writable_ds(self, ds,name, path):
        return datasets.BufferedWriteableDS(ds,name,path)

    def predict(self, dataset,target:datasets.WriteableDataSet, fold_num=0,stage=-1):
        return self.stages[stage].predict(dataset,target,fold_num)


    def predictions(self,dataset,fold_num,stage):
        pred = os.path.join(os.path.dirname(self.path), "predictions")
        ensure(pred)
        fName = os.path.join(pred,dataset + "." + str(fold_num) + "." + str(stage))

        if dataset=="validation" or dataset=="holdout" or dataset=="train":
            stage_obj=self.stages[stage]
            dsn = stage_obj.get_dataset()
            split = self.get_split(dsn, fold_num)
            if dataset=="validation":
                rd=split.validation
            if dataset=="train":
                rd=split.train
            if dataset=="holdout":
                rd=split.test
        else:
            rd=datasets.create(dataset)

        target=self.create_writable_ds(rd,dataset,fName)
        if os.path.exists(fName) or os.path.exists(fName+".npy"):
            target.load()
            return target

        self.predict(rd,target,fold_num,stage)
        target.commit()
        return target

class TrainingStage:

    def __init__(self,cfg:GenericConfig,stage_num,stage_dict):
        self.stage_dict=stage_dict
        self.num=stage_num
        self.batchSize=16
        self.dataset=None
        self.optimizer=key_or_default("optimizer",stage_dict,cfg.optimizer)
        self.callbacks=None
        self.lr=0.0001
        self.cfg=cfg
        self.stage_dict["Sparent"]=cfg.cfg_dict

        self.epochs=stage_dict['epochs']
        self.loss=key_or_default("loss",stage_dict,cfg.loss)
        self.outputTransform = cfg.outputTransform
        self.schedule=key_or_default("schedule",stage_dict,cfg.schedule)
        self.n_saved=key_or_default("n_saved",stage_dict,1)
        self.early_stopping=key_or_default("early_stopping",stage_dict,cfg.early_stopping)
        pass

    def get_dataset(self):
        return datasets.create(utils.either(self.dataset, self.cfg.dataset))

    def predict(self,dataset,targetDataset:datasets.WriteableDataSet,fold_num=0):
        mdl=self.load_model(fold_num);
        if isinstance(dataset,str):
            dataset = datasets.create(dataset)
        loader = self.cfg.create_loader(dataset)
        if torch.cuda.is_available():
            mdl = mdl.cuda()
        evaluator = create_supervised_evaluator(mdl, prepare_batch=self.cfg.batchPrepare)
        @evaluator.on(Events.ITERATION_COMPLETED)
        def accept(eng):
            batch_predictions=eng.state.output[0].cpu().numpy()
            #print(batch_predictions)
            for v in batch_predictions:
                targetDataset.append(v)
            pass
        evaluator.run(loader)
        pass



    def load_model(self,fold_num):
        ec = ExecutionConfig(fold_num, self.num, dr=os.path.dirname(self.cfg.path))
        weights_dir=os.path.dirname(ec.weightsPath())
        prefix=os.path.basename(ec.weightsPath())
        max_score=0
        best=None
        for f in os.listdir(weights_dir):
            if prefix in f:
                vl=f.index('=')
                val=f[vl+1:-4]
                score=float(val)
                if score>max_score:
                    max_score=score
                    best=f
        if best is not None:
            mdl_state=torch.load(os.path.join(weights_dir,best))
            model:torch.nn.Module=self.cfg.create_model()
            model.load_state_dict(mdl_state)
            return model
        return None

    def createOptimizer(self,model):
        originlOpt= binding_platform.create_extension(torch.optim.Optimizer,self.optimizer,model.parameters())
        if self.schedule is not None:

            self.scheduler= binding_platform.create_extension(torch.optim.lr_scheduler._LRScheduler,self.schedule,originlOpt)
        return originlOpt

    def fit(self,model):
        dataset =datasets.create(utils.either(self.dataset,self.cfg.dataset))
        if torch.cuda.is_available():
            model=model.cuda()

        for fold_num in range(self.cfg.folds_count):
            ec = ExecutionConfig(fold_num,self.num,dr=os.path.dirname(self.cfg.path))
            cb=callbacks.CSVLogger(ec.metricsPath())
            pb=callbacks.ProgbarLogger()
            pb.params={}
            cb.on_train_begin({})
            split=self.cfg.get_split(dataset,fold_num)
            train_loader=self.cfg.create_loader(split.train)
            val_loader = self.cfg.create_loader(split.validation)
            pb.params["epochs"]=self.epochs
            pb.params["verbose"]=True
            pb.params["steps"] = len(train_loader)
            pb.params["samples"]=len(train_loader)*self.batchSize
            pb.params["size"] = self.batchSize
            pb.on_train_begin({})
            optimizer = self.createOptimizer(model)
            #####
            loss = losses.create(self.loss)

            trainer = create_supervised_trainer(model, optimizer, loss, prepare_batch=self.cfg.batchPrepare)
            primary_metric = self.cfg.primary_metric
            primary_metric_mode = self.cfg.primary_metric_mode
            def score_function(engine):
                primary=engine.state.metrics[primary_metric]
                if primary_metric_mode=="min":
                    primary=-primary
                return primary
            def fname(e,v1):
                return "best"
            to_save = {'model': model}
            fp=os.path.basename(ec.weightsPath())
            saver = Checkpoint(to_save, DiskSaver(os.path.dirname(ec.weightsPath()), require_empty=False,create_dir=True), n_saved=self.n_saved,
                                 filename_prefix=fp, score_function=score_function, score_name=primary_metric,
                               global_step_transform=fname)
            if self.early_stopping is not None:
                earlyStop=callbacks.EarlyStopping(primary_metric,mode=self.cfg.primary_metric_mode,verbose=1,patience=self.early_stopping)
                earlyStop.model=trainer
                earlyStop.on_train_begin()
            else:
                earlyStop=None
            evaluator = create_supervised_evaluator(model,
                                                    metrics={
                                                        'binary_accuracy': Accuracy(),
                                                        'loss': Loss(loss)
                                                    },prepare_batch=self.cfg.batchPrepare,output_transform=self.outputTransform)
            @trainer.on(Events.ITERATION_COMPLETED)
            def log_training_loss(trainer):
                pb.params["metrics"]={"loss":trainer.state.output}
                pb.on_batch_end(trainer.state.iteration,{"loss":trainer.state.output})

            @trainer.on(Events.EPOCH_STARTED)
            def epoch_started(trainer):
                pb.on_epoch_begin(trainer.state.epoch,{})

            @trainer.on(Events.GET_BATCH_STARTED)
            def epoch_started(trainer):
                pb.on_batch_begin(trainer.state.iteration, {})

            @trainer.on(Events.EPOCH_COMPLETED)
            def log_results(trainer):
                final_metrics = {}
                evaluator.run(train_loader)
                for k in evaluator.state.metrics:
                    final_metrics[k]=evaluator.state.metrics[k]
                evaluator.run(val_loader)
                for k in evaluator.state.metrics:
                    final_metrics["val_"+k]=evaluator.state.metrics[k]
                cb.on_epoch_end(trainer.state.epoch,final_metrics)
                pb.on_epoch_end(trainer.state.epoch, final_metrics)



                currentMetricValue=final_metrics[primary_metric];
                if primary_metric_mode=="max":
                    currentMetricValue=-currentMetricValue
                if self.scheduler is not None:
                    self.scheduler.step(currentMetricValue)
                if earlyStop is not None:
                    earlyStop.on_epoch_end(trainer.state.epoch,final_metrics)
                evaluator.state.metrics=final_metrics
                saver(evaluator)
                #trainer.should_terminate=True
            #trainer.add_event_handler(Events.COMPLETED)
            trainer.run(train_loader, max_epochs=self.epochs)
            earlyStop.on_train_end()
        pass

class GenericPipeline(GenericConfig):

    def create_loader(self,ds):
        return  datasets.create_loader(ds,self.batch_size,self.num_workers,False,False)

def parse(path,extra=None)->GenericConfig:
    extraImports=[]
    if isinstance(path, str):
        if not os.path.exists(path) or os.path.isdir(path):
            cur_project_pth=context.get_current_project_path()
            config_path = os.path.join(path,"config.yaml")
            if os.path.exists(config_path):
                path=config_path
            else:
                config_path = os.path.join(cur_project_pth,"experiments",path,"config.yaml")
                if os.path.exists(config_path):
                    path=config_path            
            if os.path.exists(cur_project_pth+"/common.yaml") and extra is None:
                extra=cur_project_pth+"/common.yaml"
            modules_path = os.path.join(cur_project_pth,'modules')
            if os.path.exists(modules_path):
                if not modules_path in sys.path:
                    sys.path.insert(0, modules_path)
                for m in os.listdir(modules_path):

                    if ".py" in m:
                        importlib.import_module(m[0:-3])
                        #extraImports.append(m[0:m.index(".py")])
    cfg = configloader.parse("generic", path,extra)
    cfg.path = path    
    for e in extraImports:
        cfg.imports.append(e)
    return cfg