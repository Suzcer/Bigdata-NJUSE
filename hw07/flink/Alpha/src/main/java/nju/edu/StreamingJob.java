/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nju.edu;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

/**
 * Skeleton for a Flink Streaming Job.
 *
 * <p>For a tutorial how to write a Flink streaming application, check the
 * tutorials and examples on the <a href="http://flink.apache.org/docs/stable/">Flink Website</a>.
 *
 * <p>To package your application into a JAR file for execution, run
 * 'mvn clean package' on the command line.
 *
 * <p>If you change the name of the main class (with the public static void main(String[] args))
 * method, change the respective entry in the POM.xml file (simply search for 'mainClass').
 */
public class StreamingJob {

    //	public static void main(String[] args) throws Exception {
//		// set up the batch execution environment
//		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
//
//		/*
//		 * Here, you can start creating your execution plan for Flink.
//		 *
//		 * Start with getting some data from the environment, like
//		 * 	env.readTextFile(textPath);
//		 *
//		 * then, transform the resulting DataSet<String> using operations
//		 * like
//		 * 	.filter()
//		 * 	.flatMap()
//		 * 	.join()
//		 * 	.coGroup()
//		 *
//		 * and many more.
//		 * Have a look at the programming guide for the Java API:
//		 *
//		 * http://flink.apache.org/docs/latest/apis/batch/index.html
//		 *
//		 * and the examples
//		 *
//		 * http://flink.apache.org/docs/latest/apis/batch/examples.html
//		 *
//		 */
//
//		// execute program
//		env.execute("Flink Batch Java API Skeleton");
//	}
    public static void main(String[] args) throws Exception {
        String input = null;
        String output = null;
        ParameterTool params = ParameterTool.fromArgs(args);
        try {
            input = params.getRequired("input");
            output = params.getRequired("output");
        } catch (RuntimeException e) {
            System.out.println("Argument Error");
            e.printStackTrace();
            return;
        }
        ExecutionEnvironment env =
                ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        DataSet<String> text = env.readTextFile(input);
        DataSet<Tuple2<String, Integer>> counts = text.flatMap(new
                BatchJob.Tokenizer()).groupBy(0).sum(1);
        counts.writeAsText(output, FileSystem.WriteMode.OVERWRITE);
        env.execute("Flink Batch Java API Skeleton");
    }

    public static class Tokenizer implements FlatMapFunction<String,
            Tuple2<String, Integer>> {
        @Override
        public void flatMap(String value, Collector<Tuple2<String, Integer>>
                out) throws Exception {
//			String[] tokens = value.toLowerCase().split("\\W+");
            String[] tokens = value.toLowerCase().split("");
            for (String token : tokens) {
                if (token.length() > 0 && token.charAt(0) > 'a' && token.charAt(0) < 'z') {
//				if (token.length() > 0) {
                    System.out.println(token);
                    out.collect(new Tuple2<String, Integer>(token, 1));
                }
            }
        }
    }
}
