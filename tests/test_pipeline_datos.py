import unittest

import torch

import pipeline_datos as pdatos


class TestPipelineDatos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = pdatos.cargar_samples()
        cls.particiones = pdatos.crear_particiones(cls.samples, fold_validation=0)

    def test_particiones_sin_solape(self):
        train = set(self.particiones.train.patient_id)
        validation = set(self.particiones.validation.patient_id)
        test = set(self.particiones.test.patient_id)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)

    def test_fold_validation_correcto(self):
        self.assertEqual(set(self.particiones.validation.fold), {0})
        self.assertNotIn(0, set(self.particiones.train.fold))
        self.assertEqual(set(self.particiones.test.fold), {-1})

    def test_lectura_imagen(self):
        dataset = pdatos.BreastDCESliceDataset(self.particiones.validation.head(1))
        item = dataset[0]
        self.assertEqual(tuple(item["image"].shape), (3, 256, 256))
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertGreaterEqual(float(item["image"].min()), 0)
        self.assertLessEqual(float(item["image"].max()), 1)

    def test_flip_conjunto(self):
        imagen = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
        transform = pdatos.RandomHorizontalFlipJoint(probability=1.0)
        self.assertTrue(torch.equal(transform(imagen), torch.flip(imagen, dims=(-1,))))

    def test_estandarizacion_compartida(self):
        imagen = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
        transform = pdatos.SharedStandardize(mean=1.0, std=2.0)
        esperado = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])
        self.assertTrue(torch.allclose(transform(imagen), esperado))


if __name__ == "__main__":
    unittest.main()
