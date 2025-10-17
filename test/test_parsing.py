import sys
import os

# Ajoute le dossier 'src' à sys.path si ce n'est pas déjà fait
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import src.rag.parsing as pars


raw_json = ['''
                {
                    "query": "AAPL",
                    "filter": "and(gt(sentiment, 0.8), gt(\\"moddate\\", \\"2024-01-01\\"))"
                }
            ''',
            '''
            {
                "query": "AAPL",
                "filter": "and(eq(\\"extension\\", \\"pdf\\"), gt(sentiment(), 0.8), gt(to_date(\\"creationdate\\", 'YYYY-MM-DD'), to_date('2024-01-01', 'YYYY-MM-DD')))"
            }
            ''',
            '''
            {
                "query": "AAPL",
                "filter": "and(eq(\\"extension\\", \\"pdf\\"), gt(sentiment(), /0.8), gt(to_date(\\"creationdate\\", 'YYYY-MM-DD'), to_date('2024-01-01', 'YYYY-MM-DD')))"
            }
            ''',
            '''
            {
                "query": "AAPL",
                "filter": `and(eq(\\"extension\\", \\"pdf\\"), gt(sentiment(), /0.8), gt(to_date(\\"creationdate\\", 'YYYY-MM-DD'), to_date('2024-01-01', 'YYYY-MM-DD')))`
            }
            ''']



def test_correction_guillement(index):
        # Exemple d'utilisation
    
    print("[TEST] Entrée:",raw_json[index])
    parsed = pars.mega_json_corrector(raw_json[index])
    print(f"REUSSSIEEE : {parsed}\n")

if __name__ == "__main__":
    # for i,a in enumerate (raw_json):
    #     test_correction_guillement(i)


    test_correction_guillement(3)