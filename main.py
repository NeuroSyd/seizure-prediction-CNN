import json
import os
import os.path
import numpy as np
import keras
keras.backend.set_image_data_format('channels_first')
print ('Using Keras image_data_format=%s' % keras.backend.image_data_format())

from utils.load_signals import PrepData
from utils.prep_data import train_val_loo_split, train_val_test_split
from models.cnn import ConvNN

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def main(dataset='Kaggle2014Pred', build_type='cv'):
    print ('Main')
    with open('SETTINGS_%s.json' %dataset) as f:
        settings = json.load(f)

    makedirs(str(settings['cachedir']))
    makedirs(str(settings['resultdir']))

    if settings['dataset']=='Kaggle2014Pred':
        targets = [
            'Dog_1',
            'Dog_2',
            'Dog_3',
            'Dog_4',
            'Dog_5',
            'Patient_1',
            'Patient_2'
        ]
    elif settings['dataset']=='FB':
        targets = [
            '1',
            '3',
            #'4',
            #'5',
            '6',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21'
        ]
    else:
        targets = [
            # '1',
            # '2',
            # '3',
            # '5',
            # '9',
            # '10',
            # '13',
            # '14',
            # '18',
            # '19',
            '20',
            '21',
            '23'
        ]

    for target in targets:
        ictal_X, ictal_y = \
            PrepData(target, type='ictal', settings=settings).apply()
        interictal_X, interictal_y = \
            PrepData(target, type='interictal', settings=settings).apply()

        if build_type=='cv':
            loo_folds = train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25)
            ind = 1
            for X_train, y_train, X_val, y_val, X_test, y_test in loo_folds:
                print (X_train.shape, y_train.shape,
                       X_val.shape, y_val.shape,
                       X_test.shape, y_test.shape)

                model = ConvNN(target,batch_size=32,nb_classes=2,epochs=50,mode=build_type)
                model.setup(X_train.shape)
                model.fit(X_train, y_train, X_val, y_val)
                model.evaluate(X_test, y_test)
                # write out predictions for preictal and interictal segments
                # preictal
                X_test_p = X_test[y_test==1]
                y_test_p = model.predict_proba(X_test_p)
                filename = os.path.join(
                    str(settings['resultdir']), 'preictal_%s_%d.csv' %(target, ind))
                lines = []
                lines.append('preictal')
                for i in range(len(y_test_p)):
                    lines.append('%.4f' % ((y_test_p[i][1])))
                with open(filename, 'w') as f:
                    print >> f, '\n'.join(lines)
                print 'wrote', filename

                # interictal
                X_test_i = X_test[y_test==0]
                y_test_i = model.predict_proba(X_test_i)
                filename = os.path.join(
                    str(settings['resultdir']), 'interictal_%s_%d.csv' %(target, ind))
                lines = []
                lines.append('interictal')
                for i in range(len(y_test_i)):
                    lines.append('%.4f' % ((y_test_i[i][1])))
                with open(filename, 'w') as f:
                    print >> f, '\n'.join(lines)
                print 'wrote', filename

                ind += 1
        elif build_type=='test':
            X_train, y_train, X_val, y_val, X_test, y_test = \
                train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)
            model = ConvNN(target,batch_size=32,nb_classes=2,epochs=100,mode=build_type)
            model.setup(X_train.shape)
            #model.fit(X_train, y_train)
            fn_weights = "weights_%s_%s.h5" %(target, build_type)
            if os.path.exists(fn_weights):
                model.load_trained_weights(fn_weights)
            else:
                model.fit(X_train, y_train, X_val, y_val)
            model.evaluate(X_test, y_test)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="cv or test. cv is for leave-one-out cross-validation")
    parser.add_argument("--dataset", help="FB, CHBMIT or Kaggle2014Pred")
    args = parser.parse_args()
    assert args.mode in ['cv','test']
    main(dataset=args.dataset, build_type=args.mode)

