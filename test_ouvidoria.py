import unittest
from unittest.mock import patch
import numpy as np
from ouvidoria import carregar_base, detectar_duplicatas, dividir, projetar

class OuvidoriaTests(unittest.TestCase):
    def test_base_demonstrativa(self):
        df = carregar_base()
        self.assertEqual(len(df),40)
        self.assertEqual((df.texto.str.len()>500).sum(),5)
        self.assertTrue(df.texto.str.len().between(50,800).all())

    def test_pares_sem_diagonal_e_limiar_estrito(self):
        vetores = np.array([[1.,0.],[1.,0.],[0.,1.]])
        with patch('ouvidoria.codificar',return_value=vetores):
            self.assertEqual(detectar_duplicatas(['a','b','c'],.85),[(0,1,1.)])
            self.assertEqual(detectar_duplicatas(['a','b','c'],1),[])
        self.assertEqual(detectar_duplicatas([]),[])

    def test_chunking_e_limites(self):
        texto = 'A rua está alagada. A água invade as casas. Precisamos de limpeza. ' * 20
        partes = dividir(texto,100,20)
        self.assertGreater(len(partes),1)
        self.assertTrue(all(0<len(c)<=100 for c in partes))
        with self.assertRaises(ValueError):
            dividir(texto,100,100)
        self.assertEqual(dividir(''),[])

    def test_projecao_amostra_pequena(self):
        self.assertEqual(projetar(np.array([[1.,0.,0.]])).shape,(1,2))

if __name__ == '__main__':
    unittest.main()
