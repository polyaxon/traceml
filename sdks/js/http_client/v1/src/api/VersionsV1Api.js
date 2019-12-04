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
    define(['ApiClient', 'model/V1LogHandler', 'model/V1Versions'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS-like environments that support module.exports, like Node.
    module.exports = factory(require('../ApiClient'), require('../model/V1LogHandler'), require('../model/V1Versions'));
  } else {
    // Browser globals (root is window)
    if (!root.PolyaxonSdk) {
      root.PolyaxonSdk = {};
    }
    root.PolyaxonSdk.VersionsV1Api = factory(root.PolyaxonSdk.ApiClient, root.PolyaxonSdk.V1LogHandler, root.PolyaxonSdk.V1Versions);
  }
}(this, function(ApiClient, V1LogHandler, V1Versions) {
  'use strict';

  /**
   * VersionsV1 service.
   * @module api/VersionsV1Api
   * @version 1.0.0
   */

  /**
   * Constructs a new VersionsV1Api. 
   * @alias module:api/VersionsV1Api
   * @class
   * @param {module:ApiClient} [apiClient] Optional API client implementation to use,
   * default to {@link module:ApiClient#instance} if unspecified.
   */
  var exports = function(apiClient) {
    this.apiClient = apiClient || ApiClient.instance;


    /**
     * Callback function to receive the result of the getLogHandler operation.
     * @callback module:api/VersionsV1Api~getLogHandlerCallback
     * @param {String} error Error message, if any.
     * @param {module:model/V1LogHandler} data The data returned by the service call.
     * @param {String} response The complete HTTP response.
     */

    /**
     * List archived runs for user
     * @param {module:api/VersionsV1Api~getLogHandlerCallback} callback The callback function, accepting three arguments: error, data, response
     * data is of type: {@link module:model/V1LogHandler}
     */
    this.getLogHandler = function(callback) {
      var postBody = null;


      var pathParams = {
      };
      var queryParams = {
      };
      var collectionQueryParams = {
      };
      var headerParams = {
      };
      var formParams = {
      };

      var authNames = ['ApiKey'];
      var contentTypes = ['application/json'];
      var accepts = ['application/json'];
      var returnType = V1LogHandler;

      return this.apiClient.callApi(
        '/api/v1/log_handler', 'GET',
        pathParams, queryParams, collectionQueryParams, headerParams, formParams, postBody,
        authNames, contentTypes, accepts, returnType, callback
      );
    }

    /**
     * Callback function to receive the result of the getVersions operation.
     * @callback module:api/VersionsV1Api~getVersionsCallback
     * @param {String} error Error message, if any.
     * @param {module:model/V1Versions} data The data returned by the service call.
     * @param {String} response The complete HTTP response.
     */

    /**
     * List bookmarked runs for user
     * @param {module:api/VersionsV1Api~getVersionsCallback} callback The callback function, accepting three arguments: error, data, response
     * data is of type: {@link module:model/V1Versions}
     */
    this.getVersions = function(callback) {
      var postBody = null;


      var pathParams = {
      };
      var queryParams = {
      };
      var collectionQueryParams = {
      };
      var headerParams = {
      };
      var formParams = {
      };

      var authNames = ['ApiKey'];
      var contentTypes = ['application/json'];
      var accepts = ['application/json'];
      var returnType = V1Versions;

      return this.apiClient.callApi(
        '/api/v1/version', 'GET',
        pathParams, queryParams, collectionQueryParams, headerParams, formParams, postBody,
        authNames, contentTypes, accepts, returnType, callback
      );
    }
  };

  return exports;
}));
