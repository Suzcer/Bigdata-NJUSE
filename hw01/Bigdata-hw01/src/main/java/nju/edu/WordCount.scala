package nju.edu


import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.MapView
import scala.io.Source


object WordCount {
  /*
   *
   */
  def main(args: Array[String]) = {

    // setup link
    val conf = new SparkConf().setMaster("local").setAppName("WordCount")
    val sc = new SparkContext(conf)

    // read lines
    var lines = sc.textFile("test.txt")

    //read words
    val words = lines.flatMap(line => line.split(" "))

    //groups
    val groups = words.groupBy(word => word.toLowerCase())

    //_map
    var _map = groups.map(x => (x._1, x._2.size))

    //ans
    val ret = _map.collect().sortBy(sortRule).reverse
    ret.foreach(println)

    sc.stop()
  }

  def sortRule(tmp: (String, Int)): (Int, String) = {
    (tmp._2, tmp._1)
  }

}
