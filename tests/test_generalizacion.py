"""Evaluación separada, bolsas completas, pérdida estable y reanudación real."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

spec=importlib.util.spec_from_file_location("ent_generalizacion",Path(__file__).parents[1]/"04_entrenamiento.py")
ent=importlib.util.module_from_spec(spec); sys.modules[spec.name]=ent; spec.loader.exec_module(ent)


def roster():
    rows=[]
    for fold in range(5):
        for label in (0,1):
            for person in range(10):
                patient=f"p{fold}{label}{person:02d}"
                for index in range(1+person%3):
                    sample=f"{patient}_z{index:03d}"
                    row=dict(patient_id=patient,sample_id=sample,fold=fold,pCR=label,split="train",dataset="spy1",slice_index=index)
                    for phase in ("PRE","EARLY","LATE"):
                        row["path_"+phase.lower()]=str(Path("dataset/train")/patient/f"{sample}_{phase}.png")
                    rows.append(row)
    return pd.DataFrame(rows)


class TinyCNN(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.head=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.ReLU(),torch.nn.Dropout(.35),torch.nn.Linear(8,1))
    def forward(self,x):
        return self.head(x.mean((-2,-1)))


class TestGeneralizacion(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_particiones_deterministas_completas_y_sin_solape(self):
        samples=roster(); seen=[]
        for fold in range(5):
            roles,plan=ent.particion_generalizacion(samples,fold)
            ids={r:set(f.patient_id) for r,f in roles.items()}
            ent.validate_roles(ids,set(samples.patient_id))
            self.assertEqual(ids["evaluation"],set(samples.loc[samples.fold.eq(fold),"patient_id"]))
            _,again=ent.particion_generalizacion(samples.sample(frac=1,random_state=20),fold)
            pd.testing.assert_frame_equal(plan,again)
            seen.extend(ids["evaluation"])
        self.assertEqual(len(seen),len(set(seen)))
        with self.assertRaisesRegex(ValueError,"Fuga"):
            ent.validate_roles({"fit":{"a"},"selection":{"a"}},{"a"})

    def test_bce_paciente_equivale_a_bce_de_media_y_tiene_gradientes(self):
        z=torch.tensor([-2.,1.,.2,-.4,2.],requires_grad=True)
        y=torch.tensor([1.,1.,0.,0.,0.]); sizes=[2,3]; weight=2.4
        expected=torch.nn.functional.binary_cross_entropy(torch.stack([z[:2].sigmoid().mean(),z[2:].sigmoid().mean()]),
            torch.tensor([1.,0.]),weight=torch.tensor([weight,1.]))
        actual=ent.BCEPaciente(weight)(z,y,sizes)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        actual.backward(); self.assertTrue(torch.isfinite(z.grad).all()); self.assertTrue((z.grad!=0).all())
        extreme=torch.tensor([1000.,-1000.,-1000.,1000.],requires_grad=True)
        loss=ent.BCEPaciente(weight)(extreme,torch.tensor([0.,0.,1.,1.]),[2,2])
        loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(extreme.grad).all())
        single=ent.BCEPaciente(weight)(z.detach(),y,[1]*5)
        reference=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))(z.detach(),y)
        self.assertTrue(torch.allclose(single,reference))

    def test_bolsas_orden_y_ausencia_de_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples=roster().sort_values(["patient_id","slice_index"]).reset_index(drop=True)
            class Cache:
                def imagen(self,sample):
                    value=int(sample[-3:]);return torch.full((3,256,256),value,dtype=torch.uint8)
            loader=ent.make_loader(samples,Path(tmp),64,0,torch.Generator(),cache=Cache(),patient_batch_size=6)
            ids=[]; observed=[]
            for batch in loader:
                self.assertEqual(sum(batch["bag_sizes"]),len(batch["image"]))
                self.assertLessEqual(len(batch["bag_sizes"]),6)
                ids.extend(batch["sample_id"]);observed.extend(batch["image"][:,0,0,0].tolist())
            self.assertEqual(ids,samples.sample_id.tolist());self.assertEqual(observed,samples.slice_index.tolist())

    def test_entrena_solo_fit_selection_y_reanuda_identico(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data"; (root/"metadata").mkdir(parents=True)
            samples=roster(); samples.to_csv(root/"metadata/samples.csv",index=False)
            rng=np.random.default_rng(18)
            for row in samples.itertuples():
                for column in ent.pdatos.COLUMNAS_RUTA:
                    path=root/getattr(row,column);path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(np.full((256,256),rng.integers(15,240),dtype=np.uint8)).save(path)
            cfg=ent.configuraciones_tamano(2,2,64,0)[0];cfg["name"]="bce_paciente"
            cfg["training"].update(patient_batch_size=6,loss_unit="patient",cpu_threads=2)
            cfg["evaluation"]["protocol"]="nested_holdout"
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            roles,_=ent.particion_generalizacion(samples,0)
            allowed=set(roles["fit"].patient_id)|set(roles["selection"].patient_id)
            access=[]; original=ent.DatasetEntrenamiento.__getitem__
            def tracked(dataset,index):
                item=original(dataset,index);access.append(item["patient_id"])
                self.assertIn(item["patient_id"],allowed)
                return item
            with patch.object(ent,"CNN",TinyCNN),patch.object(ent.DatasetEntrenamiento,"__getitem__",tracked),patch.object(ent,"graficar_historia"):
                ent.run_training(cfg,"weighted",42,0,root,a,torch.device("cpu"))
                original_save=ent.save_checkpoint
                def interrupted(*args,**kwargs):
                    original_save(*args,**kwargs)
                    if args[0].name=="last.pt" and args[5]==0: raise RuntimeError("interrupción real")
                with patch.object(ent,"save_checkpoint",side_effect=interrupted),self.assertRaisesRegex(RuntimeError,"interrupción"):
                    ent.run_training(cfg,"weighted",42,0,root,b,torch.device("cpu"))
                ent.run_training(cfg,"weighted",42,0,root,b,torch.device("cpu"))
            self.assertTrue(access)
            relative=Path("bce_paciente/weighted/seed_42/fold_0")
            ca=torch.load(a/relative/"last.pt",weights_only=False);cb=torch.load(b/relative/"last.pt",weights_only=False)
            for name,value in ca["state_dict"].items(): self.assertTrue(torch.equal(value,cb["state_dict"][name]),name)
            for key in ("torch","loader","augmentation"): self.assertTrue(torch.equal(ca["rng"][key],cb["rng"][key]))
            pd.testing.assert_frame_equal(pd.read_csv(a/relative/"history.csv").drop(columns="epoch_seconds"),
                                          pd.read_csv(b/relative/"history.csv").drop(columns="epoch_seconds"))
            pd.testing.assert_frame_equal(pd.read_csv(a/relative/"selection_slices.csv"),pd.read_csv(b/relative/"selection_slices.csv"))
            config=json.loads((b/relative/"config.json").read_text())
            expected=ent.pdatos.pos_weight_cortes(roles["fit"])
            self.assertEqual(config["pos_weight"],expected)

    def test_evaluacion_congela_todas_las_decisiones_antes_de_outer(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp); samples=roster(); calls=[]
            configs=ent.configuraciones_tamano(2,2,64,0)
            for config,name in zip(configs,("referencia_cortes","bce_paciente")):
                config["name"]=name
                for fold in range(5):
                    for seed in (42,2026):
                        folder=output/"ejecuciones"/name/"weighted"/f"seed_{seed}"/f"fold_{fold}"
                        folder.mkdir(parents=True)
                        (folder/"inference.pt").write_bytes(f"fixture-{name}-{fold}-{seed}".encode())
            def predictions(config,seed,fold,role,rows,*unused):
                roles,_=ent.particion_generalizacion(samples,fold)
                self.assertEqual(set(rows.sample_id),set(roles[role].sample_id))
                if role=="evaluation":
                    self.assertTrue((output/"decisiones_congeladas.json").exists())
                    self.assertEqual(sum(r=="calibration" for r in calls),20)
                    locked=json.loads((output/"decisiones_congeladas.json").read_text())
                    self.assertEqual(sum(len(f) for f in locked["arms"].values()),10)
                calls.append(role)
                result=rows[["sample_id","patient_id","pCR","fold","dataset"]].copy()
                result["prob"]=.2+.3*rows.pCR+rows.slice_index*.01
                return result
            original=ent.bootstrap_generalizacion
            with patch.object(ent,"predicciones_rol",side_effect=predictions),patch.object(ent,"graficar_generalizacion"),patch.object(ent,"bootstrap_generalizacion",side_effect=lambda p:original(p,20)):
                result=ent.evaluar_generalizacion(samples,output,output,torch.device("cpu"),configs,None)
                self.assertEqual(result["status"],"complete")
                self.assertEqual(result["test_images_loaded"],0)
                oof=pd.read_csv(output/"oof_pacientes.csv")
                self.assertEqual(len(oof),samples.patient_id.nunique())
                path=output/"decisiones_congeladas.json"
                changed=json.loads(path.read_text());changed["arms"]["bce_paciente"]["0"]["threshold"]+=.01
                ent.json_write(path,changed)
                with self.assertRaisesRegex(ValueError,"Decisiones modificadas"):
                    ent.evaluar_generalizacion(samples,output,output,torch.device("cpu"),configs,None)

    def test_metricas_usan_umbral_congelado_de_cada_fold(self):
        y=np.array([0,1,0,1]);p=np.array([.2,.3,.6,.9]);t=np.array([.25,.25,.8,.8])
        metrics=ent.metricas_con_umbral(y,p,t)
        self.assertEqual(metrics["f1"],1.0);self.assertEqual(metrics["specificity"],1.0)
        self.assertEqual(metrics["roc_auc"],ent.roc_auc_score(y,p))


if __name__=="__main__": unittest.main()
