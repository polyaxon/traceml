// Copyright 2018-2020 Polyaxon, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Polyaxon SDKs and REST API specification.
 * Polyaxon SDKs and REST API specification.
 *
 * The version of the OpenAPI document: 1.2.0-rc3
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


package org.openapitools.client.model;

import java.util.Objects;
import java.util.Arrays;
import com.google.gson.TypeAdapter;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.annotations.SerializedName;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.openapitools.client.model.SparkDeployMode;
import org.openapitools.client.model.V1SparkReplica;
import org.openapitools.client.model.V1SparkType;

/**
 * V1Spark
 */

public class V1Spark {
  public static final String SERIALIZED_NAME_KIND = "kind";
  @SerializedName(SERIALIZED_NAME_KIND)
  private String kind = "spark";

  public static final String SERIALIZED_NAME_CONNECTIONS = "connections";
  @SerializedName(SERIALIZED_NAME_CONNECTIONS)
  private List<String> connections = null;

  public static final String SERIALIZED_NAME_VOLUMES = "volumes";
  @SerializedName(SERIALIZED_NAME_VOLUMES)
  private List<Object> volumes = null;

  public static final String SERIALIZED_NAME_TYPE = "type";
  @SerializedName(SERIALIZED_NAME_TYPE)
  private V1SparkType type = V1SparkType.JAVA;

  public static final String SERIALIZED_NAME_SPARK_VERSION = "spark_version";
  @SerializedName(SERIALIZED_NAME_SPARK_VERSION)
  private String sparkVersion;

  public static final String SERIALIZED_NAME_PYTHON_VERSION = "python_version";
  @SerializedName(SERIALIZED_NAME_PYTHON_VERSION)
  private String pythonVersion;

  public static final String SERIALIZED_NAME_DEPLOY_MODE = "deploy_mode";
  @SerializedName(SERIALIZED_NAME_DEPLOY_MODE)
  private SparkDeployMode deployMode = SparkDeployMode.CLUSTER;

  public static final String SERIALIZED_NAME_MAIN_CLASS = "main_class";
  @SerializedName(SERIALIZED_NAME_MAIN_CLASS)
  private String mainClass;

  public static final String SERIALIZED_NAME_MAIN_APPLICATION_FILE = "main_application_file";
  @SerializedName(SERIALIZED_NAME_MAIN_APPLICATION_FILE)
  private String mainApplicationFile;

  public static final String SERIALIZED_NAME_ARGUMENTS = "arguments";
  @SerializedName(SERIALIZED_NAME_ARGUMENTS)
  private List<String> arguments = null;

  public static final String SERIALIZED_NAME_HADOOP_CONF = "hadoop_conf";
  @SerializedName(SERIALIZED_NAME_HADOOP_CONF)
  private Map<String, String> hadoopConf = null;

  public static final String SERIALIZED_NAME_SPARK_CONF = "spark_conf";
  @SerializedName(SERIALIZED_NAME_SPARK_CONF)
  private Map<String, String> sparkConf = null;

  public static final String SERIALIZED_NAME_SPARK_CONFIG_MAP = "spark_config_map";
  @SerializedName(SERIALIZED_NAME_SPARK_CONFIG_MAP)
  private String sparkConfigMap;

  public static final String SERIALIZED_NAME_HADOOP_CONFIG_MAP = "hadoop_config_map";
  @SerializedName(SERIALIZED_NAME_HADOOP_CONFIG_MAP)
  private String hadoopConfigMap;

  public static final String SERIALIZED_NAME_EXECUTOR = "executor";
  @SerializedName(SERIALIZED_NAME_EXECUTOR)
  private V1SparkReplica executor;

  public static final String SERIALIZED_NAME_DRIVER = "driver";
  @SerializedName(SERIALIZED_NAME_DRIVER)
  private V1SparkReplica driver;


  public V1Spark kind(String kind) {
    
    this.kind = kind;
    return this;
  }

   /**
   * Get kind
   * @return kind
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public String getKind() {
    return kind;
  }


  public void setKind(String kind) {
    this.kind = kind;
  }


  public V1Spark connections(List<String> connections) {
    
    this.connections = connections;
    return this;
  }

  public V1Spark addConnectionsItem(String connectionsItem) {
    if (this.connections == null) {
      this.connections = new ArrayList<String>();
    }
    this.connections.add(connectionsItem);
    return this;
  }

   /**
   * Get connections
   * @return connections
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public List<String> getConnections() {
    return connections;
  }


  public void setConnections(List<String> connections) {
    this.connections = connections;
  }


  public V1Spark volumes(List<Object> volumes) {
    
    this.volumes = volumes;
    return this;
  }

  public V1Spark addVolumesItem(Object volumesItem) {
    if (this.volumes == null) {
      this.volumes = new ArrayList<Object>();
    }
    this.volumes.add(volumesItem);
    return this;
  }

   /**
   * Volumes is a list of volumes that can be mounted.
   * @return volumes
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "Volumes is a list of volumes that can be mounted.")

  public List<Object> getVolumes() {
    return volumes;
  }


  public void setVolumes(List<Object> volumes) {
    this.volumes = volumes;
  }


  public V1Spark type(V1SparkType type) {
    
    this.type = type;
    return this;
  }

   /**
   * Get type
   * @return type
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public V1SparkType getType() {
    return type;
  }


  public void setType(V1SparkType type) {
    this.type = type;
  }


  public V1Spark sparkVersion(String sparkVersion) {
    
    this.sparkVersion = sparkVersion;
    return this;
  }

   /**
   * Spark version is the version of Spark the application uses.
   * @return sparkVersion
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "Spark version is the version of Spark the application uses.")

  public String getSparkVersion() {
    return sparkVersion;
  }


  public void setSparkVersion(String sparkVersion) {
    this.sparkVersion = sparkVersion;
  }


  public V1Spark pythonVersion(String pythonVersion) {
    
    this.pythonVersion = pythonVersion;
    return this;
  }

   /**
   * Spark version is the version of Spark the application uses.
   * @return pythonVersion
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "Spark version is the version of Spark the application uses.")

  public String getPythonVersion() {
    return pythonVersion;
  }


  public void setPythonVersion(String pythonVersion) {
    this.pythonVersion = pythonVersion;
  }


  public V1Spark deployMode(SparkDeployMode deployMode) {
    
    this.deployMode = deployMode;
    return this;
  }

   /**
   * Get deployMode
   * @return deployMode
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public SparkDeployMode getDeployMode() {
    return deployMode;
  }


  public void setDeployMode(SparkDeployMode deployMode) {
    this.deployMode = deployMode;
  }


  public V1Spark mainClass(String mainClass) {
    
    this.mainClass = mainClass;
    return this;
  }

   /**
   * MainClass is the fully-qualified main class of the Spark application. This only applies to Java/Scala Spark applications.
   * @return mainClass
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "MainClass is the fully-qualified main class of the Spark application. This only applies to Java/Scala Spark applications.")

  public String getMainClass() {
    return mainClass;
  }


  public void setMainClass(String mainClass) {
    this.mainClass = mainClass;
  }


  public V1Spark mainApplicationFile(String mainApplicationFile) {
    
    this.mainApplicationFile = mainApplicationFile;
    return this;
  }

   /**
   * MainFile is the path to a bundled JAR, Python, or R file of the application.
   * @return mainApplicationFile
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "MainFile is the path to a bundled JAR, Python, or R file of the application.")

  public String getMainApplicationFile() {
    return mainApplicationFile;
  }


  public void setMainApplicationFile(String mainApplicationFile) {
    this.mainApplicationFile = mainApplicationFile;
  }


  public V1Spark arguments(List<String> arguments) {
    
    this.arguments = arguments;
    return this;
  }

  public V1Spark addArgumentsItem(String argumentsItem) {
    if (this.arguments == null) {
      this.arguments = new ArrayList<String>();
    }
    this.arguments.add(argumentsItem);
    return this;
  }

   /**
   * Arguments is a list of arguments to be passed to the application.
   * @return arguments
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "Arguments is a list of arguments to be passed to the application.")

  public List<String> getArguments() {
    return arguments;
  }


  public void setArguments(List<String> arguments) {
    this.arguments = arguments;
  }


  public V1Spark hadoopConf(Map<String, String> hadoopConf) {
    
    this.hadoopConf = hadoopConf;
    return this;
  }

  public V1Spark putHadoopConfItem(String key, String hadoopConfItem) {
    if (this.hadoopConf == null) {
      this.hadoopConf = new HashMap<String, String>();
    }
    this.hadoopConf.put(key, hadoopConfItem);
    return this;
  }

   /**
   * HadoopConf carries user-specified Hadoop configuration properties as they would use the  the \&quot;--conf\&quot; option in spark-submit.  The SparkApplication controller automatically adds prefix \&quot;spark.hadoop.\&quot; to Hadoop configuration properties.
   * @return hadoopConf
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "HadoopConf carries user-specified Hadoop configuration properties as they would use the  the \"--conf\" option in spark-submit.  The SparkApplication controller automatically adds prefix \"spark.hadoop.\" to Hadoop configuration properties.")

  public Map<String, String> getHadoopConf() {
    return hadoopConf;
  }


  public void setHadoopConf(Map<String, String> hadoopConf) {
    this.hadoopConf = hadoopConf;
  }


  public V1Spark sparkConf(Map<String, String> sparkConf) {
    
    this.sparkConf = sparkConf;
    return this;
  }

  public V1Spark putSparkConfItem(String key, String sparkConfItem) {
    if (this.sparkConf == null) {
      this.sparkConf = new HashMap<String, String>();
    }
    this.sparkConf.put(key, sparkConfItem);
    return this;
  }

   /**
   * HadoopConf carries user-specified Hadoop configuration properties as they would use the  the \&quot;--conf\&quot; option in spark-submit.  The SparkApplication controller automatically adds prefix \&quot;spark.hadoop.\&quot; to Hadoop configuration properties.
   * @return sparkConf
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "HadoopConf carries user-specified Hadoop configuration properties as they would use the  the \"--conf\" option in spark-submit.  The SparkApplication controller automatically adds prefix \"spark.hadoop.\" to Hadoop configuration properties.")

  public Map<String, String> getSparkConf() {
    return sparkConf;
  }


  public void setSparkConf(Map<String, String> sparkConf) {
    this.sparkConf = sparkConf;
  }


  public V1Spark sparkConfigMap(String sparkConfigMap) {
    
    this.sparkConfigMap = sparkConfigMap;
    return this;
  }

   /**
   * SparkConfigMap carries the name of the ConfigMap containing Spark configuration files such as log4j.properties. The controller will add environment variable SPARK_CONF_DIR to the path where the ConfigMap is mounted to.
   * @return sparkConfigMap
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "SparkConfigMap carries the name of the ConfigMap containing Spark configuration files such as log4j.properties. The controller will add environment variable SPARK_CONF_DIR to the path where the ConfigMap is mounted to.")

  public String getSparkConfigMap() {
    return sparkConfigMap;
  }


  public void setSparkConfigMap(String sparkConfigMap) {
    this.sparkConfigMap = sparkConfigMap;
  }


  public V1Spark hadoopConfigMap(String hadoopConfigMap) {
    
    this.hadoopConfigMap = hadoopConfigMap;
    return this;
  }

   /**
   * HadoopConfigMap carries the name of the ConfigMap containing Hadoop configuration files such as core-site.xml. The controller will add environment variable HADOOP_CONF_DIR to the path where the ConfigMap is mounted to.
   * @return hadoopConfigMap
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "HadoopConfigMap carries the name of the ConfigMap containing Hadoop configuration files such as core-site.xml. The controller will add environment variable HADOOP_CONF_DIR to the path where the ConfigMap is mounted to.")

  public String getHadoopConfigMap() {
    return hadoopConfigMap;
  }


  public void setHadoopConfigMap(String hadoopConfigMap) {
    this.hadoopConfigMap = hadoopConfigMap;
  }


  public V1Spark executor(V1SparkReplica executor) {
    
    this.executor = executor;
    return this;
  }

   /**
   * Get executor
   * @return executor
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public V1SparkReplica getExecutor() {
    return executor;
  }


  public void setExecutor(V1SparkReplica executor) {
    this.executor = executor;
  }


  public V1Spark driver(V1SparkReplica driver) {
    
    this.driver = driver;
    return this;
  }

   /**
   * Get driver
   * @return driver
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public V1SparkReplica getDriver() {
    return driver;
  }


  public void setDriver(V1SparkReplica driver) {
    this.driver = driver;
  }


  @Override
  public boolean equals(java.lang.Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    V1Spark v1Spark = (V1Spark) o;
    return Objects.equals(this.kind, v1Spark.kind) &&
        Objects.equals(this.connections, v1Spark.connections) &&
        Objects.equals(this.volumes, v1Spark.volumes) &&
        Objects.equals(this.type, v1Spark.type) &&
        Objects.equals(this.sparkVersion, v1Spark.sparkVersion) &&
        Objects.equals(this.pythonVersion, v1Spark.pythonVersion) &&
        Objects.equals(this.deployMode, v1Spark.deployMode) &&
        Objects.equals(this.mainClass, v1Spark.mainClass) &&
        Objects.equals(this.mainApplicationFile, v1Spark.mainApplicationFile) &&
        Objects.equals(this.arguments, v1Spark.arguments) &&
        Objects.equals(this.hadoopConf, v1Spark.hadoopConf) &&
        Objects.equals(this.sparkConf, v1Spark.sparkConf) &&
        Objects.equals(this.sparkConfigMap, v1Spark.sparkConfigMap) &&
        Objects.equals(this.hadoopConfigMap, v1Spark.hadoopConfigMap) &&
        Objects.equals(this.executor, v1Spark.executor) &&
        Objects.equals(this.driver, v1Spark.driver);
  }

  @Override
  public int hashCode() {
    return Objects.hash(kind, connections, volumes, type, sparkVersion, pythonVersion, deployMode, mainClass, mainApplicationFile, arguments, hadoopConf, sparkConf, sparkConfigMap, hadoopConfigMap, executor, driver);
  }


  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class V1Spark {\n");
    sb.append("    kind: ").append(toIndentedString(kind)).append("\n");
    sb.append("    connections: ").append(toIndentedString(connections)).append("\n");
    sb.append("    volumes: ").append(toIndentedString(volumes)).append("\n");
    sb.append("    type: ").append(toIndentedString(type)).append("\n");
    sb.append("    sparkVersion: ").append(toIndentedString(sparkVersion)).append("\n");
    sb.append("    pythonVersion: ").append(toIndentedString(pythonVersion)).append("\n");
    sb.append("    deployMode: ").append(toIndentedString(deployMode)).append("\n");
    sb.append("    mainClass: ").append(toIndentedString(mainClass)).append("\n");
    sb.append("    mainApplicationFile: ").append(toIndentedString(mainApplicationFile)).append("\n");
    sb.append("    arguments: ").append(toIndentedString(arguments)).append("\n");
    sb.append("    hadoopConf: ").append(toIndentedString(hadoopConf)).append("\n");
    sb.append("    sparkConf: ").append(toIndentedString(sparkConf)).append("\n");
    sb.append("    sparkConfigMap: ").append(toIndentedString(sparkConfigMap)).append("\n");
    sb.append("    hadoopConfigMap: ").append(toIndentedString(hadoopConfigMap)).append("\n");
    sb.append("    executor: ").append(toIndentedString(executor)).append("\n");
    sb.append("    driver: ").append(toIndentedString(driver)).append("\n");
    sb.append("}");
    return sb.toString();
  }

  /**
   * Convert the given object to string with each line indented by 4 spaces
   * (except the first line).
   */
  private String toIndentedString(java.lang.Object o) {
    if (o == null) {
      return "null";
    }
    return o.toString().replace("\n", "\n    ");
  }

}

