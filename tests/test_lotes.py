"""Composición pareada de lotes: exposición, aumentos y reanudación reales."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from test_generalizacion import roster, TinyCNN

spec=importlib.util.spec_from_file_location("ent_lotes",Path(__file__).parents[1]/"04_entrenamiento.py")
ent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ent;spec.loader.exec_module(ent)


class TestLotes(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_mismos_cortes_longitudes_pasos_y_rng_con_diversidad_distinta(self):
        samples=roster().sort_values(["patient_id","slice_index"]).reset_index(drop=True)
        a=ent.LotesPareados(samples,torch.Generator().manual_seed(42),"pacientes_completas")
        b=ent.LotesPareados(samples,torch.Generator().manual_seed(42),"cortes_mezclados")
        for epoch in range(3):
            x,y=list(a),list(b)
            self.assertEqual([len(v) for v in x],[len(v) for v in y])
            self.assertEqual(sorted(i for batch in x for i in batch),list(range(len(samples))))
            self.assertEqual(sorted(i for batch in y for i in batch),list(range(len(samples))))
            self.assertTrue(torch.equal(a.generator.get_state(),b.generator.get_state()))
            self.assertLessEqual(a.last_audit["max_distinct_patients"],6)
            self.assertGreater(b.last_audit["mean_distinct_patients"],a.last_audit["mean_distinct_patients"])
            self.assertEqual(a.last_audit["batch_sizes_sha256"],b.last_audit["batch_sizes_sha256"])
            self.assertEqual(a.last_audit["sample_membership_sha256"],b.last_audit["sample_membership_sha256"])

    def test_rechaza_muestras_test_duplicadas_y_orden_invalido(self):
        samples=roster().sort_values(["patient_id","slice_index"]).reset_index(drop=True)
        for bad in (samples.assign(split="test"),pd.concat([samples,samples.iloc[:1]]),samples.iloc[::-1]):
            with self.assertRaises(ValueError):
                ent.LotesPareados(bad,torch.Generator(),"cortes_mezclados")

    def test_aumento_es_identico_por_muestra_al_reordenar_y_alinea_fases(self):
        ids=[f"sample_{i}" for i in range(12)]
        x=torch.arange(12*3*8*8).reshape(12,3,8,8).float()
        normal=ent.aumentar_pareado(x.clone(),ids,"42/0/0")
        order=torch.randperm(12,generator=torch.Generator().manual_seed(9))
        changed=ent.aumentar_pareado(x[order].clone(),[ids[i] for i in order],"42/0/0")
        self.assertTrue(torch.equal(normal[order],changed))
        self.assertTrue(torch.equal(normal[:,1]-normal[:,0],torch.full((12,8,8),64.)))
        self.assertNotEqual(ent.codigos_aumento_pareado(ids,"42/0/0"),ent.codigos_aumento_pareado(ids,"42/0/1"))

    def test_configs_solo_difieren_politica_y_nombre(self):
        a,b=ent.configuraciones_lotes()
        for c in (a,b): c.pop("name");c["training"].pop("batch_policy")
        self.assertEqual(a,b)
        self.assertEqual(a["model"]["channels"],[24,48,96,160])
        self.assertEqual(a["training"]["loss_unit"],"slice")

    def test_reanudacion_real_y_comprobacion_del_pareado(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data";(root/"metadata").mkdir(parents=True)
            samples=roster();samples.to_csv(root/"metadata/samples.csv",index=False)
            for number,row in enumerate(samples.itertuples()):
                for column in ent.pdatos.COLUMNAS_RUTA:
                    path=root/getattr(row,column);path.parent.mkdir(parents=True,exist_ok=True)
                    pixels=np.full((256,256),15+number%210,dtype=np.uint8)
                    pixels[:128,:128]+=10
                    Image.fromarray(pixels).save(path)
            configs=ent.configuraciones_lotes(2,2,0)
            for c in configs:c["training"]["cpu_threads"]=2
            output=Path(tmp)/"paired";resumed=Path(tmp)/"resumed"
            roles,_=ent.particion_generalizacion(samples,0)
            allowed=set(roles["fit"].patient_id)|set(roles["selection"].patient_id)
            original_get=ent.DatasetEntrenamiento.__getitem__
            def tracked(dataset,index):
                value=original_get(dataset,index)
                self.assertIn(value["patient_id"],allowed)
                return value
            with patch.object(ent,"CNN",TinyCNN),patch.object(ent,"graficar_historia"),patch.object(ent.DatasetEntrenamiento,"__getitem__",tracked):
                for config in configs:
                    ent.run_training(config,"weighted",42,0,root,output/"ejecuciones",torch.device("cpu"))
                original_save=ent.save_checkpoint
                def crash(*args,**kwargs):
                    original_save(*args,**kwargs)
                    if args[0].name=="last.pt" and args[5]==0:raise RuntimeError("interrupción real")
                with patch.object(ent,"save_checkpoint",side_effect=crash),self.assertRaisesRegex(RuntimeError,"interrupción"):
                    ent.run_training(configs[1],"weighted",42,0,root,resumed,torch.device("cpu"))
                ent.run_training(configs[1],"weighted",42,0,root,resumed,torch.device("cpu"))
            report=ent.verificar_lotes_pareados(output,configs,seeds=(42,),folds=(0,))
            self.assertEqual(report["status"],"passed")
            relative=Path("cortes_mezclados/weighted/seed_42/fold_0")
            a=torch.load(output/"ejecuciones"/relative/"last.pt",weights_only=False)
            b=torch.load(resumed/relative/"last.pt",weights_only=False)
            for name,value in a["state_dict"].items():self.assertTrue(torch.equal(value,b["state_dict"][name]),name)
            for key in ("torch","loader","augmentation"):self.assertTrue(torch.equal(a["rng"][key],b["rng"][key]),key)
            for epoch in (1,2):
                self.assertEqual(json.loads((output/"ejecuciones"/relative/f"lotes_{epoch:03d}.json").read_text()),
                                 json.loads((resumed/relative/f"lotes_{epoch:03d}.json").read_text()))
            audit_path=output/"ejecuciones"/relative/"lotes_001.json"
            changed=json.loads(audit_path.read_text());changed["augmentation_assignment_sha256"]="changed"
            ent.json_write(audit_path,changed)
            with self.assertRaisesRegex(ValueError,"augmentation_assignment"):
                ent.verificar_lotes_pareados(output,configs,seeds=(42,),folds=(0,))


if __name__=="__main__":unittest.main()
