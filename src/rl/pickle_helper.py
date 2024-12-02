

import pickle


class PickleHelper:

    @staticmethod
    def pickle_safedump(fp: str, data: any) -> bool:

        try:
            with open(fp, 'wb') as f:
                pickle.dump(obj=data, file=f)
            return True
        
        except Exception as e:
            print(f'[E] Exception for pickle dump : {e}')
            return False
        
    @staticmethod
    def pickle_safeload(fp: str) -> any:

        try:
            with open(fp, 'rb') as f:
                obj: any = pickle.load(file=f)
            return obj
        
        except Exception as e:
            print(f'[E] Exception for pickle load : {e}')


