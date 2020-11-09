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
 *
 */

import ApiClient from '../ApiClient';
import V1Environment from './V1Environment';
import V1Init from './V1Init';

/**
 * The V1Job model module.
 * @module model/V1Job
 * @version 1.3.1
 */
class V1Job {
    /**
     * Constructs a new <code>V1Job</code>.
     * @alias module:model/V1Job
     */
    constructor() { 
        
        V1Job.initialize(this);
    }

    /**
     * Initializes the fields of this object.
     * This method is used by the constructors of any subclasses, in order to implement multiple inheritance (mix-ins).
     * Only for internal use.
     */
    static initialize(obj) { 
    }

    /**
     * Constructs a <code>V1Job</code> from a plain JavaScript object, optionally creating a new instance.
     * Copies all relevant properties from <code>data</code> to <code>obj</code> if supplied or a new instance if not.
     * @param {Object} data The plain JavaScript object bearing properties of interest.
     * @param {module:model/V1Job} obj Optional instance to populate.
     * @return {module:model/V1Job} The populated <code>V1Job</code> instance.
     */
    static constructFromObject(data, obj) {
        if (data) {
            obj = obj || new V1Job();

            if (data.hasOwnProperty('kind')) {
                obj['kind'] = ApiClient.convertToType(data['kind'], 'String');
            }
            if (data.hasOwnProperty('environment')) {
                obj['environment'] = V1Environment.constructFromObject(data['environment']);
            }
            if (data.hasOwnProperty('connections')) {
                obj['connections'] = ApiClient.convertToType(data['connections'], ['String']);
            }
            if (data.hasOwnProperty('volumes')) {
                obj['volumes'] = ApiClient.convertToType(data['volumes'], [Object]);
            }
            if (data.hasOwnProperty('init')) {
                obj['init'] = ApiClient.convertToType(data['init'], [V1Init]);
            }
            if (data.hasOwnProperty('sidecars')) {
                obj['sidecars'] = ApiClient.convertToType(data['sidecars'], [Object]);
            }
            if (data.hasOwnProperty('container')) {
                obj['container'] = ApiClient.convertToType(data['container'], Object);
            }
        }
        return obj;
    }


}

/**
 * @member {String} kind
 * @default 'job'
 */
V1Job.prototype['kind'] = 'job';

/**
 * @member {module:model/V1Environment} environment
 */
V1Job.prototype['environment'] = undefined;

/**
 * @member {Array.<String>} connections
 */
V1Job.prototype['connections'] = undefined;

/**
 * Volumes is a list of volumes that can be mounted.
 * @member {Array.<Object>} volumes
 */
V1Job.prototype['volumes'] = undefined;

/**
 * @member {Array.<module:model/V1Init>} init
 */
V1Job.prototype['init'] = undefined;

/**
 * @member {Array.<Object>} sidecars
 */
V1Job.prototype['sidecars'] = undefined;

/**
 * @member {Object} container
 */
V1Job.prototype['container'] = undefined;






export default V1Job;

