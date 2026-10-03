"""Contratos del PDF, separación de pacientes y reanudación por época."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

spec = importlib.util.spec_from_file_location("entrenamiento", Path(__file__).parents[1]/"04_entrenamiento.py")
entrenamiento = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrenamiento)


def datos_sinteticos(root, imagenes=False, folds=(0,1,2)):
    (root/"metadata").mkdir(parents=True)
    rows, rng = [], np.random.default_rng(10)
    for fold in folds:
        for label in (0, 1):
            patient = f"p{fold}{label}"
            for index in (0, 1):
                sample = f"{patient}_z{index:03d}"
                row = {"sample_id": sample, "patient_id": patient, "fold": fold, "pCR": label,
                       "split": "train", "dataset": "spy1", "slice_index": index}
                for phase in ("PRE", "EARLY", "LATE"):
                    relative = Path("dataset/train")/patient/f"{sample}_{phase}.png"
                    row[f"path_{phase.lower()}"] = str(relative)
                    if imagenes:
                        path = root/relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(rng.integers(0, 255, (256,256), dtype=np.uint8)).save(path)
                rows.append(row)
    row = rows[0].copy()
    row.update(sample_id="test_z000",patient_id="test",split="test",fold=-1,pCR="NO_LEER")
    rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(root/"metadata/samples.csv",index=False)
    return frame


class TestEntrenamiento(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_arquitectura_del_pdf_y_gradientes(self):
        model = entrenamiento.CNN()
        self.assertEqual(entrenamiento.numero_parametros(model),551913)
        self.assertEqual([sum(p.numel() for p in b.parameters()) for b in model.features],
                         [5928,31296,124800,369280])
        shapes=[]
        handles=[b.register_forward_hook(lambda m,i,o: shapes.append(tuple(o.shape[1:])))
                 for b in model.features]
        result=model(torch.rand(2,3,256,256))
        self.assertEqual(shapes,[(24,64,64),(48,32,32),(96,16,16),(160,8,8)])
        self.assertEqual(tuple(result.shape),(2,1))
        torch.nn.BCEWithLogitsLoss()(result[:,0],torch.tensor([0.,1.])).backward()
        self.assertGreater(float(model.features[0][0].weight.grad.abs().sum()),0)
        for h in handles:
            h.remove()

    def test_train_no_convierte_etiquetas_test_y_valida_folds(self):
        frame=datos_sinteticos(self.root)
        samples=entrenamiento.cargar_train(self.root)
        self.assertEqual(set(samples.split),{"train"})
        tr,va=entrenamiento.particion(samples,0)
        self.assertFalse(set(tr.patient_id)&set(va.patient_id))
        self.assertEqual(entrenamiento.pdatos.pos_weight_cortes(tr),1)
        broken=pd.concat([frame,frame.iloc[[0]].assign(sample_id="p00_z002",fold=1)])
        broken.to_csv(self.root/"metadata/samples.csv",index=False)
        with self.assertRaisesRegex(ValueError,"fold inconsistente"):
            entrenamiento.cargar_train(self.root)

    def test_rechaza_paciente_compartida_con_test(self):
        frame=datos_sinteticos(self.root)
        frame.loc[frame.split.eq("test"),"patient_id"]="p00"
        frame.to_csv(self.root/"metadata/samples.csv",index=False)
        with self.assertRaisesRegex(ValueError,"Fuga por paciente"):
            entrenamiento.cargar_train(self.root)

    def test_pipeline_escala_una_vez_y_conserva_fases(self):
        datos_sinteticos(self.root)
        samples=entrenamiento.cargar_train(self.root).head(1)
        for column,value in zip(entrenamiento.pdatos.COLUMNAS_RUTA,(32,96,64)):
            path=self.root/getattr(samples.iloc[0],column)
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.full((256,256),value,np.uint8)).save(path)
        dataset=entrenamiento.DatasetEntrenamiento(samples,self.root)
        batch=next(iter(entrenamiento.make_loader(samples,self.root,1,0,torch.Generator())))
        x,y=entrenamiento.batch_device(batch,torch.device("cpu"))
        np.testing.assert_allclose(x[0,:,0,0].numpy(),np.array([32,96,64])/255,rtol=1e-6)
        self.assertEqual(tuple(dataset[0]["image"].shape),(3,256,256))
        altered=samples.copy()
        altered["path_pre"]="dataset/test/p00/p00_z000_PRE.png"
        with self.assertRaisesRegex(ValueError,"Ruta/fase"):
            entrenamiento.DatasetEntrenamiento(altered,self.root)

    def test_aumentos_no_desalinean_canales(self):
        base=torch.arange(256*256).reshape(1,1,256,256).float()
        image=torch.cat([base,base+100000,base+200000],dim=1).repeat(6,1,1,1)
        result=entrenamiento.aumentar_geometria(image,torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(result[:,1]-result[:,0],torch.full((6,256,256),100000.)))
        self.assertTrue(torch.equal(result[:,2]-result[:,0],torch.full((6,256,256),200000.)))

    def test_oof_completo_y_por_paciente(self):
        datos_sinteticos(self.root)
        roster=entrenamiento.cargar_train(self.root).rename(columns={"pCR":"label","dataset":"cohort"})
        oof=roster.copy()
        oof["probability"]=np.linspace(.1,.9,len(oof))
        valid=entrenamiento.validate_oof(oof,roster)
        self.assertEqual(len(entrenamiento.aggregate_patients(valid)),6)
        with self.assertRaisesRegex(ValueError,"incompleto"):
            entrenamiento.validate_oof(oof.iloc[:-1],roster)
        broken=oof.copy()
        broken.loc[0,"fold"]=4
        with self.assertRaisesRegex(ValueError,"fold"):
            entrenamiento.validate_oof(broken,roster)
        with self.assertRaisesRegex(ValueError,"test"):
            entrenamiento.validate_oof(oof.assign(split="test"),roster)

    def test_calibracion_y_umbral_del_pdf(self):
        calibration={"slope":.6680108289790444,"intercept":-.815361445727788}
        p=entrenamiento.apply_calibration(np.array([.6]),calibration)[0]
        self.assertAlmostEqual(p,.367,places=3)
        self.assertGreater(p,.2982406880601241)
        for constant in (0.,.2,.5,.8,1.):
            self.assertEqual(entrenamiento.youden_threshold(np.array([0,1,0,1]),np.full(4,constant)),.5)

    def test_reanudacion_real_es_identica(self):
        datos_sinteticos(self.root,imagenes=True)
        config=entrenamiento.configuracion()
        config["model"].update(channels=[2,2,2,2],hidden=4)
        config["training"].update(epochs=3,min_epochs=3,batch_size=2,num_workers=0,
                                   cpu_threads=2,amp=False)
        a,b=self.root/"continua",self.root/"reanudada"
        args=(config,"weighted",42,0,self.root)
        entrenamiento.run_training(*args,a,torch.device("cpu"))
        original=entrenamiento.history_write
        def interrumpir(path,history):
            original(path,history)
            if len(history)==1:
                raise KeyboardInterrupt("Interrupción simulada tras last.pt")
        with patch.object(entrenamiento,"history_write",interrumpir):
            with self.assertRaises(KeyboardInterrupt):
                entrenamiento.run_training(*args,b,torch.device("cpu"))
        entrenamiento.run_training(*args,b,torch.device("cpu"))
        relative=Path("raw_rot90/weighted/seed_42/fold_0")
        ca=torch.load(a/relative/"last.pt",map_location="cpu",weights_only=False)
        cb=torch.load(b/relative/"last.pt",map_location="cpu",weights_only=False)
        for name,value in ca["state_dict"].items():
            self.assertTrue(torch.equal(value,cb["state_dict"][name]),name)
        self.assertEqual(ca["scheduler"],cb["scheduler"])
        for key in ("torch","loader","augmentation"):
            self.assertTrue(torch.equal(ca["rng"][key],cb["rng"][key]),key)
        ha=pd.read_csv(a/relative/"history.csv").drop(columns=["epoch_seconds"])
        hb=pd.read_csv(b/relative/"history.csv").drop(columns=["epoch_seconds"])
        pd.testing.assert_frame_equal(ha,hb)
        pd.testing.assert_frame_equal(pd.read_csv(a/relative/"oof_slices.csv"),
                                      pd.read_csv(b/relative/"oof_slices.csv"))
        artifact=torch.load(b/relative/"inference.pt",map_location="cpu",weights_only=True)
        model=entrenamiento.CNN(artifact["model_config"])
        model.load_state_dict(artifact["state_dict"],strict=True)
        changed=json.loads(json.dumps(config))
        changed["training"]["lr"]*=2
        with self.assertRaisesRegex(ValueError,"Configuración distinta"):
            entrenamiento.run_training(changed,"weighted",42,0,self.root,b,torch.device("cpu"))

    def test_comparacion_completa_y_rechazo_de_fold_ausente(self):
        datos_sinteticos(self.root,folds=range(5))
        train=entrenamiento.cargar_train(self.root)
        output=self.root/"ejecuciones"
        signature=hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
        omitted=None
        for name in ("base_raw","raw_rot90"):
            for loss in ("normal","weighted"):
                for seed in (42,2026):
                    for fold in range(5):
                        folder=output/name/loss/f"seed_{seed}"/f"fold_{fold}"
                        folder.mkdir(parents=True)
                        config=entrenamiento.configuracion(name)
                        config.update(loss=loss,seed=seed,fold=fold,smoke=False,data_signature=signature)
                        oof=train[train.fold.eq(fold)].copy()
                        confidence=.95 if name=="raw_rot90" and loss=="weighted" else .6
                        oof["prob"]=np.where(oof.pCR.eq(1),confidence,1-confidence)
                        oof.to_csv(folder/"oof_slices.csv",index=False)
                        (folder/"inference.pt").write_bytes(b"fixture: comparison only, not loaded")
                        (folder/"config.json").write_text(json.dumps(config))
                        summary={"status":"complete","eligible_for_selection":True,
                                 "config_hash":entrenamiento.config_hash(config),
                                 "oof_sha256":entrenamiento.sha256(folder/"oof_slices.csv"),
                                 "inference_sha256":entrenamiento.sha256(folder/"inference.pt")}
                        (folder/"summary.json").write_text(json.dumps(summary))
                        omitted=folder/"summary.json"
        report=entrenamiento.comparar_ejecuciones(self.root,output)
        self.assertEqual(report["selected"]["configuration"],"raw_rot90")
        self.assertEqual(report["selected"]["loss"],"weighted")
        self.assertEqual(report["selected"]["aggregation"],"mean")
        manifest_path=output/"comparacion/modelo_desarrollo.json"
        manifest=entrenamiento.cargar_manifest(manifest_path)
        self.assertEqual(len(manifest["models"]),10)
        self.assertFalse(manifest["test_evaluated_once"])
        omitted.unlink()
        with self.assertRaisesRegex(ValueError,"cinco folds completos"):
            entrenamiento.comparar_ejecuciones(self.root,output)


if __name__ == "__main__":
    unittest.main()
