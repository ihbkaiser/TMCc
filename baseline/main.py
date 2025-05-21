from utils import config, log, miscellaneous, seed
import os
import numpy as np
import basic_trainer
from models.ECRTM.ECRTM import ECRTM
from models.FASTOPIC.FASTOPIC import FASTOPIC
from models.NSTM.NSTM import NSTM
from models.CTM import CTM
from models.ETM import ETM
from models.ProdLDA import ProdLDA
from models.WETE import WeTe
import evaluations
import datasethandler
import scipy
import torch
import wandb

RESULT_DIR = 'results'
DATA_DIR = 'datasets'

if __name__ == "__main__":
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_logging_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    config.add_wete_argument(parser)
    args = parser.parse_args()
    
    prj = args.wandb_prj if args.wandb_prj else 'baselines'

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR + "/" + str(args.model) + "/" +str(args.dataset), current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)
    
    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))
    wandb.login(key="")
    wandb.init(project=prj, config=args)
    wandb.log({'time_stamp': current_time})

    # if args.dataset in ['YahooAnswers']:
    #     read_labels = True
    # else:
    #     read_labels = False
    read_labels = True

    # load a preprocessed dataset
    dataset = datasethandler.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=True)

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()

    if args.model == "ECRTM":
        model = ECRTM(vocab_size=dataset.vocab_size, 
                      num_topics=args.num_topics, 
                      dropout=args.dropout, 
                      pretrained_WE=pretrainWE if args.use_pretrainWE else None, 
                      weight_loss_ECR=args.weight_ECR)
    elif args.model == "FASTOPIC":
        model = FASTOPIC(vocab_size=dataset.vocab_size, num_topics=args.num_topics)
    elif args.model == "NSTM":
        model = NSTM(vocab_size=dataset.vocab_size, num_topics=args.num_topics,
                     pretrained_WE=pretrainWE if args.use_pretrainWE else None)
    elif args.model == "CTM":
        model = CTM(vocab_size=dataset.vocab_size,
                    contextual_emb_size=dataset.contextual_embed_size,
                    num_topics=args.num_topics,
                    dropout=args.dropout)
    elif args.model == "ETM":
        model = ETM(vocab_size=dataset.vocab_size, 
                    num_topics=args.num_topics,
                    pretrained_WE=pretrainWE if args.use_pretrainWE else None, 
                    train_WE=True)
    elif args.model == "ProdLDA":
        model = ProdLDA(vocab_size=dataset.vocab_size, 
                    num_topics=args.num_topics,
                    dropout=args.dropout)
    elif args.model == "WeTe":
        model = WeTe(vocab_size=dataset.vocab_size, vocab=dataset.vocab, num_topics=args.num_topics,device=args.device)
    model = model.to(args.device)

    # create a trainer
    if args.model == "FASTOPIC":
        trainer = basic_trainer.FastBasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device)
    elif args.model == "WeTe":
        trainer = basic_trainer.WeteBasicTrainer(model,epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device)
    elif args.model == "CTM":
        trainer = basic_trainer.CTMBasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size)
    else:
        trainer = basic_trainer.BasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device)


    # train the model
    
    if args.model == "FASTOPIC":
        train_simple_embedding, train_theta = trainer.train(dataset)
    # save beta, theta and top words
        beta = trainer.save_beta(current_run_dir)
        test_theta = trainer.model.get_theta(dataset.test_contextual_embed, train_simple_embedding)
        train_theta = np.asarray(train_theta.cpu())
        test_theta = np.asarray(test_theta.cpu())
    else:
        trainer.train(dataset)
    # save beta, theta and top words
        beta = trainer.save_beta(current_run_dir)
        train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir)

    # argmax of train and test theta
    # train_theta_argmax = train_theta.argmax(axis=1)
    # test_theta_argmax = test_theta.argmax(axis=1) 
    train_theta_argmax = train_theta.argmax(axis=1)
    unique_elements, counts = np.unique(train_theta_argmax, return_counts=True)
    print(f'train theta argmax: {unique_elements, counts}')
    logger.info(f'train theta argmax: {unique_elements, counts}')
    test_theta_argmax = test_theta.argmax(axis=1)
    unique_elements, counts = np.unique(test_theta_argmax, return_counts=True)
    print(f'test theta argmax: {unique_elements, counts}')
    logger.info(f'test theta argmax: {unique_elements, counts}')       

    # TD_15 = evaluations.compute_topic_diversity(
    #     top_words_15, _type="TD")
    # print(f"TD_15: {TD_15:.5f}")


    # # evaluating clustering
    # if read_labels:
    #     clustering_results = evaluations.evaluate_clustering(
    #         test_theta, dataset.test_labels)
    #     print(f"NMI: ", clustering_results['NMI'])
    #     print(f'Purity: ', clustering_results['Purity'])


    # TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
    #     os.path.join(current_run_dir, 'top_words_15.txt'))
    # print(f"TC_15: {TC_15:.5f}")
    TD_10 = evaluations.compute_topic_diversity(
        top_words_10, _type="TD")
    print(f"TD_10: {TD_10:.5f}")
    wandb.log({"TD_10": TD_10})
    # logger.info(f"TD_10: {TD_10:.5f}")

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")
    wandb.log({"TD_15": TD_15})
    # logger.info(f"TD_15: {TD_15:.5f}")

    # TD_20 = topmost.evaluations.compute_topic_diversity(
    #     top_words_20, _type="TD")
    # print(f"TD_20: {TD_20:.5f}")
    # wandb.log({"TD_20": TD_20})
    # logger.info(f"TD_20: {TD_20:.5f}")

    # TD_25 = topmost.evaluations.compute_topic_diversity(
    #     top_words_25, _type="TD")
    # print(f"TD_25: {TD_25:.5f}")
    # wandb.log({"TD_25": TD_25})
    # logger.info(f"TD_25: {TD_25:.5f}")

    # evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        wandb.log({"NMI": clustering_results['NMI']})
        wandb.log({"Purity": clustering_results['Purity']})
        # logger.info(f"NMI: {clustering_results['NMI']}")
        # logger.info(f"Purity: {clustering_results['Purity']}")

    # evaluate classification
    if read_labels:
        classification_results = evaluations.evaluate_classification(
            train_theta, test_theta, dataset.train_labels, dataset.test_labels, tune=args.tune_SVM)
        print(f"Accuracy: ", classification_results['acc'])
        wandb.log({"Accuracy": classification_results['acc']})
        # logger.info(f"Accuracy: {classification_results['acc']}")
        print(f"Macro-f1", classification_results['macro-F1'])
        wandb.log({"Macro-f1": classification_results['macro-F1']})
        # logger.info(f"Macro-f1: {classification_results['macro-F1']}")

    # TC
    TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")
    wandb.log({"TC_15": TC_15})
    # logger.info(f"TC_15: {TC_15:.5f}")
    # logger.info(f'TC_15 list: {TC_15_list}')

    # TC_10_list, TC_10 = topmost.evaluations.topic_coherence.TC_on_wikipedia(
    #     os.path.join(current_run_dir, 'top_words_10.txt'))
    # print(f"TC_10: {TC_10:.5f}")
    # wandb.log({"TC_10": TC_10})
    # logger.info(f"TC_10: {TC_10:.5f}")
    # logger.info(f'TC_10 list: {TC_10_list}')

    # NPMI
    NPMI_train_10_list, NPMI_train_10 = evaluations.compute_topic_coherence(
        dataset.train_texts, dataset.vocab, top_words_10, cv_type='c_npmi')
    print(f"NPMI_train_10: {NPMI_train_10:.5f}, NPMI_train_10_list: {NPMI_train_10_list}")
    wandb.log({"NPMI_train_10": NPMI_train_10})
    # logger.info(f"NPMI_train_10: {NPMI_train_10:.5f}")
    # logger.info(f'NPMI_train_10 list: {NPMI_train_10_list}')

    NPMI_wiki_10_list, NPMI_wiki_10 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_10.txt'), cv_type='NPMI')
    print(f"NPMI_wiki_10: {NPMI_wiki_10:.5f}, NPMI_wiki_10_list: {NPMI_wiki_10_list}")
    wandb.log({"NPMI_wiki_10": NPMI_wiki_10})
    # logger.info(f"NPMI_wiki_10: {NPMI_wiki_10:.5f}")
    # logger.info(f'NPMI_wiki_10 list: {NPMI_wiki_10_list}')

    Cp_wiki_10_list, Cp_wiki_10 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_10.txt'), cv_type='C_P')
    print(f"Cp_wiki_10: {Cp_wiki_10:.5f}, Cp_wiki_10_list: {Cp_wiki_10_list}")
    wandb.log({"Cp_wiki_10": Cp_wiki_10})
    # logger.info(f"Cp_wiki_10: {Cp_wiki_10:.5f}")
    # logger.info(f'Cp_wiki_10 list: {Cp_wiki_10_list}')
    
    # w2v_list, w2v = evaluations.topic_coherence.compute_topic_coherence(
    #     dataset.train_texts, dataset.vocab, top_words_10, cv_type='c_w2v')
    # print(f"w2v: {w2v:.5f}, w2v_list: {w2v_list}")
    # wandb.log({"w2v": w2v})
    # logger.info(f"w2v: {w2v:.5f}")
    # logger.info(f'w2v list: {w2v_list}')

    wandb.finish()
