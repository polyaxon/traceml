// Copyright 2019 Polyaxon, Inc.
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
 * OpenAPI spec version: 1.0.0
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by the swagger code generator program.
 * https://github.com/swagger-api/swagger-codegen.git
 *
 * Swagger Codegen version: 2.4.10
 *
 * Do not edit the class manually.
 *
 */

(function(root, factory) {
  if (typeof define === 'function' && define.amd) {
    // AMD. Register as an anonymous module.
    define(['ApiClient', 'model/V1SearchSpec'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS-like environments that support module.exports, like Node.
    module.exports = factory(require('../ApiClient'), require('./V1SearchSpec'));
  } else {
    // Browser globals (root is window)
    if (!root.PolyaxonSdk) {
      root.PolyaxonSdk = {};
    }
    root.PolyaxonSdk.V1WidgetSpec = factory(root.PolyaxonSdk.ApiClient, root.PolyaxonSdk.V1SearchSpec);
  }
}(this, function(ApiClient, V1SearchSpec) {
  'use strict';

  /**
   * The V1WidgetSpec model module.
   * @module model/V1WidgetSpec
   * @version 1.0.0
   */

  /**
   * Constructs a new <code>V1WidgetSpec</code>.
   * @alias module:model/V1WidgetSpec
   * @class
   */
  var exports = function() {
  };

  /**
   * Constructs a <code>V1WidgetSpec</code> from a plain JavaScript object, optionally creating a new instance.
   * Copies all relevant properties from <code>data</code> to <code>obj</code> if supplied or a new instance if not.
   * @param {Object} data The plain JavaScript object bearing properties of interest.
   * @param {module:model/V1WidgetSpec} obj Optional instance to populate.
   * @return {module:model/V1WidgetSpec} The populated <code>V1WidgetSpec</code> instance.
   */
  exports.constructFromObject = function(data, obj) {
    if (data) {
      obj = obj || new exports();
      if (data.hasOwnProperty('king'))
        obj.king = ApiClient.convertToType(data['king'], 'String');
      if (data.hasOwnProperty('search'))
        obj.search = V1SearchSpec.constructFromObject(data['search']);
      if (data.hasOwnProperty('meta'))
        obj.meta = ApiClient.convertToType(data['meta'], Object);
    }
    return obj;
  }

  /**
   * @member {String} king
   */
  exports.prototype.king = undefined;

  /**
   * @member {module:model/V1SearchSpec} search
   */
  exports.prototype.search = undefined;

  /**
   * @member {Object} meta
   */
  exports.prototype.meta = undefined;

  return exports;

}));
