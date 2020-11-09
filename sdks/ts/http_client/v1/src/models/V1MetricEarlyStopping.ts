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

/* tslint:disable */
/* eslint-disable */
/**
 * Polyaxon SDKs and REST API specification.
 * Polyaxon SDKs and REST API specification.
 *
 * The version of the OpenAPI document: 1.3.1
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */

import { exists, mapValues } from '../runtime';
import {
    V1Optimization,
    V1OptimizationFromJSON,
    V1OptimizationFromJSONTyped,
    V1OptimizationToJSON,
} from './';

/**
 * MetricEarlyStoppingSchema specification
 * Early stopping based on metric config.
 * @export
 * @interface V1MetricEarlyStopping
 */
export interface V1MetricEarlyStopping {
    /**
     * 
     * @type {string}
     * @memberof V1MetricEarlyStopping
     */
    kind?: string;
    /**
     * Metric name to use for early stopping.
     * @type {string}
     * @memberof V1MetricEarlyStopping
     */
    metric?: string;
    /**
     * Metric value to use for the condition.
     * @type {string}
     * @memberof V1MetricEarlyStopping
     */
    value?: string;
    /**
     * 
     * @type {V1Optimization}
     * @memberof V1MetricEarlyStopping
     */
    optimization?: V1Optimization;
    /**
     * 
     * @type {object}
     * @memberof V1MetricEarlyStopping
     */
    policy?: object;
}

export function V1MetricEarlyStoppingFromJSON(json: any): V1MetricEarlyStopping {
    return V1MetricEarlyStoppingFromJSONTyped(json, false);
}

export function V1MetricEarlyStoppingFromJSONTyped(json: any, ignoreDiscriminator: boolean): V1MetricEarlyStopping {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'kind': !exists(json, 'kind') ? undefined : json['kind'],
        'metric': !exists(json, 'metric') ? undefined : json['metric'],
        'value': !exists(json, 'value') ? undefined : json['value'],
        'optimization': !exists(json, 'optimization') ? undefined : V1OptimizationFromJSON(json['optimization']),
        'policy': !exists(json, 'policy') ? undefined : json['policy'],
    };
}

export function V1MetricEarlyStoppingToJSON(value?: V1MetricEarlyStopping | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'kind': value.kind,
        'metric': value.metric,
        'value': value.value,
        'optimization': V1OptimizationToJSON(value.optimization),
        'policy': value.policy,
    };
}


