package nju.edu.Clustering;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;


/**
 * 设置分群数为3类，观察到KMeans方法可以正好分为3种类别的鸢尾属植物。
 * 分别是 versicolor 杂色， setosa 刚毛，virginia类的鸢尾属植物。
 * 可以观察到不同类别的一些属性特征。(sepal 萼片, petal花瓣)
 */
public class EM_ {
    public static void main(String[] args) throws Exception {
        String file = "src/main/resources/iris.arff";
        FileReader FReader = new FileReader(file);
        BufferedReader Reader = new BufferedReader(FReader);
        Instances data = new Instances(Reader);
        removeMissingData(data);

        data.setClassIndex(data.numAttributes() - 1);//设置最后一个属性作为分类属性
        Remove filter = new Remove();

        /**
         *   filter.setAttributeIndices();
         *   Set which attributes are to be deleted (or kept if invert is true)
         *   用来设置哪一个属性应该被删除的方法。参数:
         *   rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1.
         *   eg: first-3,5,6-last
         *   */
        filter.setAttributeIndices("" + (data.classIndex() + 1));

        filter.setInputFormat(data);

        Instances dataCluster = Filter.useFilter(data, filter);     //使用过滤器进行过滤，返回新的数据集

        EM clusterer = new EM();
        /**
         * Valid options are:
         * -N: number of clusters. If omitted or -1 specified, then cross validation is used to select the number of clusters.
         * -I: max iterations.(default 100)
         * -V: verbose.
         * -M: minimum allowable standard deviation for normal density computation(default 1e-6)
         * -O: Display model in old format (good when there are many clusters)
         * -S: Random number seed.(default 100)
         * */
        String[] options = new String[4];

        options[0] = "-I";
        options[1] = "100"; //设置最大迭代次数
        options[2] = "-N";
        options[3] = "3";   //设置簇的个数

        clusterer.setOptions(options);

        clusterer.buildClusterer(dataCluster);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(data);

        System.out.println("数据总数：" + data.numInstances() + "   属性个数为：" + data.numAttributes());
        System.out.println(eval.clusterResultsToString());


        /**
         * 对数似然度量,评估聚类算法的质量。
         * 数据集划分为多个折,针对每个折运行聚类。
         * 如果簇类算法簇里面的相似数据的概率高,说明它在捕获数据结构方面做得很好。
         */
        double logLikelihood = ClusterEvaluation.crossValidateModel(clusterer, dataCluster, 10, new Random(1));
        System.out.println("logLikelihood: "+logLikelihood);

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
