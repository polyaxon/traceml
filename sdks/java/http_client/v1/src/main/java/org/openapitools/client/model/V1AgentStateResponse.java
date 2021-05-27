// Copyright 2018-2021 Polyaxon, Inc.
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
 * The version of the OpenAPI document: 1.9.3
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
import org.openapitools.client.model.AgentStateResponseAgentState;
import org.openapitools.client.model.V1Statuses;

/**
 * V1AgentStateResponse
 */

public class V1AgentStateResponse {
  public static final String SERIALIZED_NAME_STATUS = "status";
  @SerializedName(SERIALIZED_NAME_STATUS)
  private V1Statuses status = V1Statuses.CREATED;

  public static final String SERIALIZED_NAME_STATE = "state";
  @SerializedName(SERIALIZED_NAME_STATE)
  private AgentStateResponseAgentState state;

  public static final String SERIALIZED_NAME_LIVE_STATE = "live_state";
  @SerializedName(SERIALIZED_NAME_LIVE_STATE)
  private Integer liveState;

  public static final String SERIALIZED_NAME_COMPATIBLE_UPDATES = "compatible_updates";
  @SerializedName(SERIALIZED_NAME_COMPATIBLE_UPDATES)
  private Object compatibleUpdates;


  public V1AgentStateResponse status(V1Statuses status) {
    
    this.status = status;
    return this;
  }

   /**
   * Get status
   * @return status
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public V1Statuses getStatus() {
    return status;
  }


  public void setStatus(V1Statuses status) {
    this.status = status;
  }


  public V1AgentStateResponse state(AgentStateResponseAgentState state) {
    
    this.state = state;
    return this;
  }

   /**
   * Get state
   * @return state
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public AgentStateResponseAgentState getState() {
    return state;
  }


  public void setState(AgentStateResponseAgentState state) {
    this.state = state;
  }


  public V1AgentStateResponse liveState(Integer liveState) {
    
    this.liveState = liveState;
    return this;
  }

   /**
   * Get liveState
   * @return liveState
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public Integer getLiveState() {
    return liveState;
  }


  public void setLiveState(Integer liveState) {
    this.liveState = liveState;
  }


  public V1AgentStateResponse compatibleUpdates(Object compatibleUpdates) {
    
    this.compatibleUpdates = compatibleUpdates;
    return this;
  }

   /**
   * Get compatibleUpdates
   * @return compatibleUpdates
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "")

  public Object getCompatibleUpdates() {
    return compatibleUpdates;
  }


  public void setCompatibleUpdates(Object compatibleUpdates) {
    this.compatibleUpdates = compatibleUpdates;
  }


  @Override
  public boolean equals(java.lang.Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    V1AgentStateResponse v1AgentStateResponse = (V1AgentStateResponse) o;
    return Objects.equals(this.status, v1AgentStateResponse.status) &&
        Objects.equals(this.state, v1AgentStateResponse.state) &&
        Objects.equals(this.liveState, v1AgentStateResponse.liveState) &&
        Objects.equals(this.compatibleUpdates, v1AgentStateResponse.compatibleUpdates);
  }

  @Override
  public int hashCode() {
    return Objects.hash(status, state, liveState, compatibleUpdates);
  }


  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class V1AgentStateResponse {\n");
    sb.append("    status: ").append(toIndentedString(status)).append("\n");
    sb.append("    state: ").append(toIndentedString(state)).append("\n");
    sb.append("    liveState: ").append(toIndentedString(liveState)).append("\n");
    sb.append("    compatibleUpdates: ").append(toIndentedString(compatibleUpdates)).append("\n");
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

