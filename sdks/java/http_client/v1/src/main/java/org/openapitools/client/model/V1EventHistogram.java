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
import java.util.List;

/**
 * V1EventHistogram
 */

public class V1EventHistogram {
  public static final String SERIALIZED_NAME_VALUES = "values";
  @SerializedName(SERIALIZED_NAME_VALUES)
  private List<Double> values = null;

  public static final String SERIALIZED_NAME_COUNTS = "counts";
  @SerializedName(SERIALIZED_NAME_COUNTS)
  private List<Double> counts = null;


  public V1EventHistogram values(List<Double> values) {
    
    this.values = values;
    return this;
  }

  public V1EventHistogram addValuesItem(Double valuesItem) {
    if (this.values == null) {
      this.values = new ArrayList<Double>();
    }
    this.values.add(valuesItem);
    return this;
  }

   /**
   * Get values
   * @return values
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public List<Double> getValues() {
    return values;
  }


  public void setValues(List<Double> values) {
    this.values = values;
  }


  public V1EventHistogram counts(List<Double> counts) {
    
    this.counts = counts;
    return this;
  }

  public V1EventHistogram addCountsItem(Double countsItem) {
    if (this.counts == null) {
      this.counts = new ArrayList<Double>();
    }
    this.counts.add(countsItem);
    return this;
  }

   /**
   * Get counts
   * @return counts
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public List<Double> getCounts() {
    return counts;
  }


  public void setCounts(List<Double> counts) {
    this.counts = counts;
  }


  @Override
  public boolean equals(java.lang.Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    V1EventHistogram v1EventHistogram = (V1EventHistogram) o;
    return Objects.equals(this.values, v1EventHistogram.values) &&
        Objects.equals(this.counts, v1EventHistogram.counts);
  }

  @Override
  public int hashCode() {
    return Objects.hash(values, counts);
  }


  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class V1EventHistogram {\n");
    sb.append("    values: ").append(toIndentedString(values)).append("\n");
    sb.append("    counts: ").append(toIndentedString(counts)).append("\n");
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

