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
/**
 * 
 * @export
 * @interface V1DateTimeSchedule
 */
export interface V1DateTimeSchedule {
    /**
     * 
     * @type {string}
     * @memberof V1DateTimeSchedule
     */
    kind?: string;
    /**
     * 
     * @type {Date}
     * @memberof V1DateTimeSchedule
     */
    start_at?: Date;
}

export function V1DateTimeScheduleFromJSON(json: any): V1DateTimeSchedule {
    return V1DateTimeScheduleFromJSONTyped(json, false);
}

export function V1DateTimeScheduleFromJSONTyped(json: any, ignoreDiscriminator: boolean): V1DateTimeSchedule {
    if ((json === undefined) || (json === null)) {
        return json;
    }
    return {
        
        'kind': !exists(json, 'kind') ? undefined : json['kind'],
        'start_at': !exists(json, 'start_at') ? undefined : (new Date(json['start_at'])),
    };
}

export function V1DateTimeScheduleToJSON(value?: V1DateTimeSchedule | null): any {
    if (value === undefined) {
        return undefined;
    }
    if (value === null) {
        return null;
    }
    return {
        
        'kind': value.kind,
        'start_at': value.start_at === undefined ? undefined : (value.start_at.toISOString()),
    };
}


