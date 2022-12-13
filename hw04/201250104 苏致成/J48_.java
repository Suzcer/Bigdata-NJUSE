package nju.edu.Classification;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;

public class J48_ {
    public static void main(String[] args) throws Exception {

        // 80%作为训练集，20%作为测试集
        double p = 20;

        File inputFile = new File("adult_income_uk.arff");
        ArffLoader loader = new ArffLoader();
        loader.setFile(inputFile);
        Instances dataSet = loader.getDataSet();
        System.out.println("原数据集大小:" + dataSet.numInstances());

        Instances newDataSet = removeMissingData(dataSet);
        newDataSet.setClassIndex(newDataSet.numAttributes() - 1);

        // Randomize data
        Randomize rand = new Randomize();
        rand.setInputFormat(newDataSet);
        rand.setRandomSeed(42);
        newDataSet = Filter.useFilter(newDataSet, rand);

        // Remove test percentage from data to get the train set
        RemovePercentage rp = new RemovePercentage();
        rp.setInputFormat(newDataSet);
        rp.setPercentage(p);
        Instances train = Filter.useFilter(newDataSet, rp);

        // Remove train percentage from data to get the test set
        rp = new RemovePercentage();
        rp.setInputFormat(newDataSet);
        rp.setPercentage(p);
        rp.setInvertSelection(true);
        Instances test = Filter.useFilter(newDataSet, rp);


        System.out.println("剔除缺失值后数据集大小:" + newDataSet.numInstances());
        System.out.println("训练集大小:" + train.numInstances());
        System.out.println("测试集大小:" + test.numInstances());

        double sum = test.numInstances();
        int right = 0;
        Classifier clas = new J48();
        clas.buildClassifier(train);
        for (int i = 0; i < sum; i++) {
            if (clas.classifyInstance(test.instance(i)) == test.instance(i).classValue()) {
                right++;
            }
//            System.out.println("预测结果："+clas.classifyInstance(insTest.instance(i))+" 真实结果：　"+insTest.instance(i).classValue());
        }
        System.out.println("NaiveBayes分类方法，分类准确率：  " + right / sum);
    }


    /**
     * 移除掉包含特殊值的属性的实例
     */
    public static Instances removeMissingData(Instances instances) {
        try {
            //logger.info("删除[{}]属性包含[{}]的实例", attribute, incompatible);
            // 属性个数（列）
            int dim = instances.numAttributes();
            // 实例个数（行）
            int num = instances.numInstances();
            for (int i = 0; i < dim; i++) {

                for (int j = 0; j < num; j++) {
                    // 实例的该属性值包含不合条件值 删除该条实例（行）
                    if (instances.instance(j).isMissing(i)) {
                        //logger.info("删除的实例属性值为[{}]", instances.instance(j).toStringNoWeight());
                        instances.remove(j);
                        j--;
                        num--;
                    }
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return instances;
    }
}
