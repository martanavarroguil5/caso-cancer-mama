"""Conservación de versiones, integridad de pesos y publicación sin copias parciales."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("versionar", Path(__file__).parents[1] / "modelos/versionar.py")
versionar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(versionar)


class TestVersiones(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "modelos"
        self.source = self.root / "resultados"
        self.source.mkdir()
        for name in versionar.CODE_FILES:
            (self.root / name).write_text("# Código compatible conservado\n", encoding="utf-8")
        models = []
        for seed in (42, 2026):
            for fold in range(5):
                path = self.source / f"seed_{seed}_fold_{fold}.pt"
                path.write_bytes(f"pesos {seed} {fold}".encode())
                models.append({"seed": seed, "fold": fold, "path": path.name, "sha256": versionar.sha256(path)})
        evidence = self.source / "metricas.json"
        evidence.write_bytes(b'{"test_original": true}\r\n')
        self.manifest = self.source / "modelo.json"
        versionar.write_json(self.manifest, {"models": models, "status": "historico_conservado",
            "selection": {"metrics_raw_oof": {"roc_auc": .6}}, "preprocessing": {},
            "provenance_files": [{"path": evidence.name, "sha256": versionar.sha256(evidence)}],
            "test_metrics_path": evidence.name})
        self.manifest.with_suffix(".json.sha256").write_text(versionar.sha256(self.manifest) + "\n")
        self.check = patch.object(versionar, "run_verification", return_value={"models": 10, "checksums_verified": True})
        self.check.start()
        self.addCleanup(self.check.stop)

    def save(self, identifier="v001_original_20261007", evidence=()):
        return versionar.archive_version(self.manifest, identifier, "Modelo conservado", evidence,
                                        archive=self.archive, source_root=self.root)

    def test_paquete_funciona_tras_desaparecer_la_fuente(self):
        row = self.save()
        for path in self.source.iterdir():
            path.unlink()
        result = versionar.verify_version(row["id"], archive=self.archive, runtime=False)
        self.assertEqual(result["models"], 10)
        self.assertTrue(result["checksums_verified"])
        folder = self.archive / "versiones" / row["id"]
        manifest = json.loads((folder / "manifiesto.json").read_text())
        self.assertEqual((folder / manifest["test_metrics_path"]).read_bytes(), b'{"test_original": true}\r\n')

    def test_no_sobrescribe_version_ni_reutiliza_numero(self):
        row = self.save()
        before = (self.archive / "versiones" / row["id"] / "version.json").read_bytes()
        for identifier in (row["id"], "v001_otro_nombre"):
            with self.assertRaises(ValueError):
                self.save(identifier)
        self.assertEqual((self.archive / "versiones" / row["id"] / "version.json").read_bytes(), before)
        self.assertEqual(len(versionar.read_versions(self.archive)), 1)

    def test_detecta_pesos_modificados(self):
        row = self.save()
        path = self.archive / "versiones" / row["id"] / "pesos/semilla_42_fold_0.pt"
        path.write_bytes(b"pesos reemplazados")
        with self.assertRaisesRegex(ValueError, "Archivo modificado"):
            versionar.verify_version(row["id"], archive=self.archive, runtime=False)

    def test_fallo_no_publica_version_parcial_y_conserva_anteriores(self):
        original = self.save()
        with self.assertRaises(FileNotFoundError):
            self.save("v002_nuevo_20261007", evidence=[self.root / "falta.json"])
        self.assertFalse((self.archive / "versiones/v002_nuevo_20261007").exists())
        self.assertFalse(list(self.archive.glob(".staging-*")))
        self.assertFalse((self.archive / ".versionar.lock").exists())
        versionar.verify_version(original["id"], archive=self.archive, runtime=False)

    def test_detecta_codigo_conservado_modificado(self):
        row = self.save()
        (self.archive / "versiones" / row["id"] / "codigo/04_entrenamiento.py").write_text("# cambio\n")
        with self.assertRaisesRegex(ValueError, "Archivo modificado"):
            versionar.verify_version(row["id"], archive=self.archive, runtime=False)

    def test_guardar_nuevo_rechaza_cambios_en_la_version_anterior(self):
        row = self.save()
        (self.archive / "versiones" / row["id"] / "pesos/semilla_42_fold_0.pt").write_bytes(b"cambio")
        with self.assertRaisesRegex(ValueError, "Archivo modificado"):
            self.save("v002_nuevo_20261007")
        self.assertFalse((self.archive / "versiones/v002_nuevo_20261007").exists())

    def test_no_admite_rutas_de_version_fuera_del_archivo(self):
        for identifier in ("../v001_original", "v000_original", "v001_original/../../fuera"):
            with self.assertRaises(ValueError):
                self.save(identifier)
        with self.assertRaises(ValueError):
            versionar.inside(self.root, "../fuera.pt")

    def test_archiva_el_paquete_actual_sin_archivos_antiguos(self):
        (self.root / "04_entrenamiento.py").unlink()
        (self.root / "pipeline_datos.py").unlink()
        package = self.root / "src/cancer_mama"
        package.mkdir(parents=True)
        for name in ("__init__.py", "entrenamiento.py", "datos.py", "paths.py"):
            (package / name).write_text("# Paquete compatible conservado\n", encoding="utf-8")
        row = self.save()
        folder = self.archive / "versiones" / row["id"]
        self.assertTrue((folder / "codigo/src/cancer_mama/entrenamiento.py").is_file())
        self.assertFalse((folder / "codigo/04_entrenamiento.py").exists())
        self.assertEqual(versionar.training_code(folder / "codigo"), folder / "codigo/src/cancer_mama/entrenamiento.py")
        versionar.verify_version(row["id"], archive=self.archive, runtime=False)


if __name__ == "__main__":
    unittest.main()
