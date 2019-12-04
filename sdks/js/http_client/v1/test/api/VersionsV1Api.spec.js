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
    // AMD.
    define(['expect.js', '../../src/index'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS-like environments that support module.exports, like Node.
    factory(require('expect.js'), require('../../src/index'));
  } else {
    // Browser globals (root is window)
    factory(root.expect, root.PolyaxonSdk);
  }
}(this, function(expect, PolyaxonSdk) {
  'use strict';

  var instance;

  beforeEach(function() {
    instance = new PolyaxonSdk.VersionsV1Api();
  });

  describe('(package)', function() {
    describe('VersionsV1Api', function() {
      describe('getLogHandler', function() {
        it('should call getLogHandler successfully', function(done) {
          // TODO: uncomment getLogHandler call and complete the assertions
          /*

          instance.getLogHandler(function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1LogHandler);
            expect(data.dsn).to.be.a('string');
            expect(data.dsn).to.be("");
            expect(data.environment).to.be.a('string');
            expect(data.environment).to.be("");

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
      describe('getVersions', function() {
        it('should call getVersions successfully', function(done) {
          // TODO: uncomment getVersions call and complete the assertions
          /*

          instance.getVersions(function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1Versions);
            expect(data.platform_version).to.be.a('string');
            expect(data.platform_version).to.be("");
            expect(data.cli).to.be.a(PolyaxonSdk.V1Version);
                  expect(data.cli.min_version).to.be.a('string');
              expect(data.cli.min_version).to.be("");
              expect(data.cli.latest_version).to.be.a('string');
              expect(data.cli.latest_version).to.be("");
            expect(data.platform).to.be.a(PolyaxonSdk.V1Version);
                  expect(data.platform.min_version).to.be.a('string');
              expect(data.platform.min_version).to.be("");
              expect(data.platform.latest_version).to.be.a('string');
              expect(data.platform.latest_version).to.be("");
            expect(data.agent).to.be.a(PolyaxonSdk.V1Version);
                  expect(data.agent.min_version).to.be.a('string');
              expect(data.agent.min_version).to.be("");
              expect(data.agent.latest_version).to.be.a('string');
              expect(data.agent.latest_version).to.be("");

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
    });
  });

}));
