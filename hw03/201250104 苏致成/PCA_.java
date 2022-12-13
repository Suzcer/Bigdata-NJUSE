package nju.edu.DimensionReduction;

import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class PCA_ {
    public static void main(String[] args) throws Exception {
        Instances data = ConverterUtils.DataSource.read("titanic.arff");
        removeMissingData(data);

        data.setClassIndex(data.numAttributes()-1);
        PrincipalComponents pca = new PrincipalComponents();
        AttributeSelection selection = new AttributeSelection();

        Ranker ranker = new Ranker();

        ranker.setNumToSelect(3);

        selection.setEvaluator(pca);

        selection.setSearch(ranker);

        selection.setInputFormat(data);
        Instances instances = Filter.useFilter(data, selection);

        System.out.println(instances);
//        pca.setCenterData(true);

//        pca.buildEvaluator(data);
        System.out.println(pca);

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
