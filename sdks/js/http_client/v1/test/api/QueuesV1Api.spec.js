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
 * The version of the OpenAPI document: 1.2.0-rc3
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 *
 */

(function(root, factory) {
  if (typeof define === 'function' && define.amd) {
    // AMD.
    define(['expect.js', process.cwd()+'/src/index'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS-like environments that support module.exports, like Node.
    factory(require('expect.js'), require(process.cwd()+'/src/index'));
  } else {
    // Browser globals (root is window)
    factory(root.expect, root.PolyaxonSdk);
  }
}(this, function(expect, PolyaxonSdk) {
  'use strict';

  var instance;

  beforeEach(function() {
    instance = new PolyaxonSdk.QueuesV1Api();
  });

  var getProperty = function(object, getter, property) {
    // Use getter method if present; otherwise, get the property directly.
    if (typeof object[getter] === 'function')
      return object[getter]();
    else
      return object[property];
  }

  var setProperty = function(object, setter, property, value) {
    // Use setter method if present; otherwise, set the property directly.
    if (typeof object[setter] === 'function')
      object[setter](value);
    else
      object[property] = value;
  }

  describe('QueuesV1Api', function() {
    describe('createQueue', function() {
      it('should call createQueue successfully', function(done) {
        //uncomment below and update the code to test createQueue
        //instance.createQueue(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('deleteQueue', function() {
      it('should call deleteQueue successfully', function(done) {
        //uncomment below and update the code to test deleteQueue
        //instance.deleteQueue(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('getQueue', function() {
      it('should call getQueue successfully', function(done) {
        //uncomment below and update the code to test getQueue
        //instance.getQueue(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('listOrganizationQueueNames', function() {
      it('should call listOrganizationQueueNames successfully', function(done) {
        //uncomment below and update the code to test listOrganizationQueueNames
        //instance.listOrganizationQueueNames(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('listOrganizationQueues', function() {
      it('should call listOrganizationQueues successfully', function(done) {
        //uncomment below and update the code to test listOrganizationQueues
        //instance.listOrganizationQueues(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('listQueueNames', function() {
      it('should call listQueueNames successfully', function(done) {
        //uncomment below and update the code to test listQueueNames
        //instance.listQueueNames(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('listQueues', function() {
      it('should call listQueues successfully', function(done) {
        //uncomment below and update the code to test listQueues
        //instance.listQueues(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('patchQueue', function() {
      it('should call patchQueue successfully', function(done) {
        //uncomment below and update the code to test patchQueue
        //instance.patchQueue(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
    describe('updateQueue', function() {
      it('should call updateQueue successfully', function(done) {
        //uncomment below and update the code to test updateQueue
        //instance.updateQueue(function(error) {
        //  if (error) throw error;
        //expect().to.be();
        //});
        done();
      });
    });
  });

}));
