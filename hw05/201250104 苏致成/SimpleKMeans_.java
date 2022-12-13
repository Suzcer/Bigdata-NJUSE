package nju.edu.Clustering;

import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * 设置分群数为3类，观察到KMeans方法可以正好分为3种类别的鸢尾属植物。
 * 分别是 versicolor 杂色， setosa 刚毛，virginia类的鸢尾属植物。
 * 可以观察到不同类别的一些属性特征。(sepal 萼片, petal花瓣)
 */
public class SimpleKMeans_ {
    public static void main(String[] args) {
        Instances dataSet = null;
        SimpleKMeans KM = null;

        try {
            // 读入样本数据
            File file = new File("src/main/resources/bmw-browsers.arff");
            ArffLoader loader = new ArffLoader();
            loader.setFile(file);
            dataSet = loader.getDataSet();
            removeMissingData(dataSet);

            // 初始化聚类器 (加载算法)

            KM = new SimpleKMeans();
            KM.setNumClusters(3);                           //设置聚类要得到的类别数量
            KM.buildClusterer(dataSet);                     //开始进行聚类
            System.out.println(KM.preserveInstancesOrderTipText());
            System.out.println(KM.toString());

        } catch(Exception e) {
            e.printStackTrace();
        }
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
