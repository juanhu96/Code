def nested_cross_validate(X, Y, estimator, c_grid, seed, model, n_splits=5, index=None, plot_DT=True, resultdir='/mnt/phd/jihu/opioid/Result/'):
    
    ## outer cv
    train_outer = []
    test_outer = []
    outer_cv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    
    for train, test in outer_cv.split(X, Y):
        train_outer.append(train)
        test_outer.append(test)
        
    ## storing lists
    best_params = []
    train_auc = []
    validation_auc = []
    auc_diffs = []
    
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    holdout_accuracy = []
    holdout_recall = []
    holdout_precision = []
    holdout_roc_auc = []
    holdout_pr_auc = []
    holdout_f1 = []
    holdout_f2 = []
    holdout_brier = []
    holdout_calibration = []
    
    ## inner cv
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i in range(len(train_outer)):
        
        ## subset train & test sets in inner loop
        train_x, test_x = X.iloc[train_outer[i]], X.iloc[test_outer[i]]
        train_y, test_y = Y[train_outer[i]], Y[test_outer[i]]
      
        '''
        ### Jingyuan: to specify grid on estimator, need to add 'estimator__' in c_grid in baseline_functions.py
        pipeline = Pipeline(steps=[('sampler', SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
                                    ('estimator', estimator)])

        clf = GridSearchCV(estimator=pipeline, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv, 
                           return_train_score=True).fit(train_x, train_y) 
        '''
        
        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=estimator, 
                            param_grid=c_grid, 
                            scoring='roc_auc',
                            cv=inner_cv, 
                            return_train_score=True).fit(train_x, train_y) 
    
        ## best parameter & scores
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        train_auc.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]])
        validation_auc.append(clf.best_score_)
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)

        ## train model on best param
        if index == 'svm':
            best_model = CalibratedClassifierCV(clf, cv=5)
            best_model.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)
            holdout_acc = best_model.score(test_x, test_y)            
        else:
            prob = clf.predict_proba(test_x)[:, 1]
            holdout_pred = clf.predict(test_x)
            holdout_acc = clf.score(test_x, test_y)
        
        ## store results
        best_params.append(best_param)
        holdout_probability.append(prob)
        holdout_prediction.append(holdout_pred)
        holdout_y.append(test_y)
        holdout_accuracy.append(accuracy_score(test_y, holdout_pred))
        holdout_recall.append(recall_score(test_y, holdout_pred))
        holdout_precision.append(precision_score(test_y, holdout_pred))
        holdout_roc_auc.append(roc_auc_score(test_y, prob))
        holdout_pr_auc.append(average_precision_score(test_y, prob))
        holdout_brier.append(brier_score_loss(test_y, prob))
        holdout_f1.append(fbeta_score(test_y, holdout_pred, beta = 1))
        holdout_f2.append(fbeta_score(test_y, holdout_pred, beta = 2))
    
        # NEW: calibration error
        holdout_calibration.append(compute_calibration(test_y, prob, holdout_pred))


        best_estimator = clf.best_estimator_
        if model == 'XGB':
            importance_scores = best_estimator.feature_importances_
            nonzero_indices = np.nonzero(importance_scores)[0]
            print(f'XGB iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'RF':
            importance_scores = best_estimator.feature_importances_
            nonzero_indices = np.nonzero(importance_scores)[0]
            print(f'RF iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'LinearSVM':
            # create a SelectFromModel object based on the trained model
            sfm = SelectFromModel(best_estimator)
            # fit the SelectFromModel object to the training data
            sfm.fit(train_x, train_y)
            # get the number of selected features
            num_selected = np.sum(sfm.get_support())
            # print('LinearSVM iteration ' + str(i) + '...' + str(num_selected) + '\n')
            print(f'LinearSVM iteration {str(i)}\n Indices: {num_selected}\n Number of features: {num_selected}\n')

        elif model == 'Lasso':
            coefficients = best_estimator.coef_[0]
            nonzero_indices = np.nonzero(coefficients)[0]
            # print('Lasso iteration ' + str(i) + '...' + str(nonzero_indices) + '\n')
            # print(coefficients)
            print(f'Lasso (L1) iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Coefficients: {coefficients}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'Logistic':
            coefficients = best_estimator.coef_[0]
            nonzero_indices = np.nonzero(coefficients)[0]
            # print('Logistic iteration ' + str(i) + '...' + str(nonzero_indices) + '\n')
            # print(coefficients)
            print(f'Logistic (L2) iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Coefficients: {coefficients}\n Number of features: {len(nonzero_indices)}\n')

        elif model == 'DT':
            
            importance_scores = best_estimator.feature_importances_
            nonzero_indices = np.nonzero(importance_scores)[0]
            # print('DT iteration ' + str(i) + '...' + str(nonzero_indices) + '\n')
            print(f'DT iteration {str(i)}\n Indices: {str(nonzero_indices)}\n Number of features: {len(nonzero_indices)}\n')
            print(best_estimator.tree_)

            # plot
            if plot_DT:
                plt.figure(figsize=(30, 30))
                tree.plot_tree(best_estimator)
                plt.savefig(f'{resultdir}DecisionTree_CV.png', dpi=300)


    return {'best_param': best_params,
            'train_auc': train_auc,
            'validation_auc': validation_auc,
            'auc_diffs': auc_diffs,
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            "holdout_test_precision": holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            "holdout_test_brier": holdout_brier,
            'holdout_test_f1': holdout_f1,
            "holdout_test_f2": holdout_f2,
            "holdout_calibration_error": holdout_calibration}