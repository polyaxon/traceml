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
    instance = new PolyaxonSdk.GitAccessesV1Api();
  });

  describe('(package)', function() {
    describe('GitAccessesV1Api', function() {
      describe('createGitAccess', function() {
        it('should call createGitAccess successfully', function(done) {
          // TODO: uncomment, update parameter values for createGitAccess call and complete the assertions
          /*
          var owner = "owner_example";
          var body = new PolyaxonSdk.V1HostAccess();
          body.uuid = "";
          body.name = "";
          body.description = "";
          body.readme = "";
          body.tags = [""];
          body.created_at = new Date();
          body.updated_at = new Date();
          body.frozen = false;
          body.disabled = false;
          body.insecure = false;
          body.deleted = false;
          body.k8s_secret = "";
          body.url = "";

          instance.createGitAccess(owner, body, function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1HostAccess);
            expect(data.uuid).to.be.a('string');
            expect(data.uuid).to.be("");
            expect(data.name).to.be.a('string');
            expect(data.name).to.be("");
            expect(data.description).to.be.a('string');
            expect(data.description).to.be("");
            expect(data.readme).to.be.a('string');
            expect(data.readme).to.be("");
            {
              let dataCtr = data.tags;
              expect(dataCtr).to.be.an(Array);
              expect(dataCtr).to.not.be.empty();
              for (let p in dataCtr) {
                let data = dataCtr[p];
                expect(data).to.be.a('string');
                expect(data).to.be("");
              }
            }
            expect(data.created_at).to.be.a(Date);
            expect(data.created_at).to.be(new Date());
            expect(data.updated_at).to.be.a(Date);
            expect(data.updated_at).to.be(new Date());
            expect(data.frozen).to.be.a('boolean');
            expect(data.frozen).to.be(false);
            expect(data.disabled).to.be.a('boolean');
            expect(data.disabled).to.be(false);
            expect(data.insecure).to.be.a('boolean');
            expect(data.insecure).to.be(false);
            expect(data.deleted).to.be.a('boolean');
            expect(data.deleted).to.be(false);
            expect(data.k8s_secret).to.be.a('string');
            expect(data.k8s_secret).to.be("");
            expect(data.url).to.be.a('string');
            expect(data.url).to.be("");

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
      describe('deleteGitAccess', function() {
        it('should call deleteGitAccess successfully', function(done) {
          // TODO: uncomment, update parameter values for deleteGitAccess call
          /*
          var owner = "owner_example";
          var uuid = "uuid_example";

          instance.deleteGitAccess(owner, uuid, function(error, data, response) {
            if (error) {
              done(error);
              return;
            }

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
      describe('getGitAccess', function() {
        it('should call getGitAccess successfully', function(done) {
          // TODO: uncomment, update parameter values for getGitAccess call and complete the assertions
          /*
          var owner = "owner_example";
          var uuid = "uuid_example";

          instance.getGitAccess(owner, uuid, function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1HostAccess);
            expect(data.uuid).to.be.a('string');
            expect(data.uuid).to.be("");
            expect(data.name).to.be.a('string');
            expect(data.name).to.be("");
            expect(data.description).to.be.a('string');
            expect(data.description).to.be("");
            expect(data.readme).to.be.a('string');
            expect(data.readme).to.be("");
            {
              let dataCtr = data.tags;
              expect(dataCtr).to.be.an(Array);
              expect(dataCtr).to.not.be.empty();
              for (let p in dataCtr) {
                let data = dataCtr[p];
                expect(data).to.be.a('string');
                expect(data).to.be("");
              }
            }
            expect(data.created_at).to.be.a(Date);
            expect(data.created_at).to.be(new Date());
            expect(data.updated_at).to.be.a(Date);
            expect(data.updated_at).to.be(new Date());
            expect(data.frozen).to.be.a('boolean');
            expect(data.frozen).to.be(false);
            expect(data.disabled).to.be.a('boolean');
            expect(data.disabled).to.be(false);
            expect(data.insecure).to.be.a('boolean');
            expect(data.insecure).to.be(false);
            expect(data.deleted).to.be.a('boolean');
            expect(data.deleted).to.be(false);
            expect(data.k8s_secret).to.be.a('string');
            expect(data.k8s_secret).to.be("");
            expect(data.url).to.be.a('string');
            expect(data.url).to.be("");

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
      describe('listGitAccessNames', function() {
        it('should call listGitAccessNames successfully', function(done) {
          // TODO: uncomment, update parameter values for listGitAccessNames call and complete the assertions
          /*
          var owner = "owner_example";
          var opts = {};
          opts.offset = 56;
          opts.limit = 56;
          opts.sort = "sort_example";
          opts.query = "query_example";

          instance.listGitAccessNames(owner, opts, function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1ListHostAccessesResponse);
            expect(data.count).to.be.a('number');
            expect(data.count).to.be(0);
            {
              let dataCtr = data.results;
              expect(dataCtr).to.be.an(Array);
              expect(dataCtr).to.not.be.empty();
              for (let p in dataCtr) {
                let data = dataCtr[p];
                expect(data).to.be.a(PolyaxonSdk.V1HostAccess);
                expect(data.uuid).to.be.a('string');
                expect(data.uuid).to.be("");
                expect(data.name).to.be.a('string');
                expect(data.name).to.be("");
                expect(data.description).to.be.a('string');
                expect(data.description).to.be("");
                expect(data.readme).to.be.a('string');
                expect(data.readme).to.be("");
                {
                  let dataCtr = data.tags;
                  expect(dataCtr).to.be.an(Array);
                  expect(dataCtr).to.not.be.empty();
                  for (let p in dataCtr) {
                    let data = dataCtr[p];
                    expect(data).to.be.a('string');
                    expect(data).to.be("");
                  }
                }
                expect(data.created_at).to.be.a(Date);
                expect(data.created_at).to.be(new Date());
                expect(data.updated_at).to.be.a(Date);
                expect(data.updated_at).to.be(new Date());
                expect(data.frozen).to.be.a('boolean');
                expect(data.frozen).to.be(false);
                expect(data.disabled).to.be.a('boolean');
                expect(data.disabled).to.be(false);
                expect(data.insecure).to.be.a('boolean');
                expect(data.insecure).to.be(false);
                expect(data.deleted).to.be.a('boolean');
                expect(data.deleted).to.be(false);
                expect(data.k8s_secret).to.be.a('string');
                expect(data.k8s_secret).to.be("");
                expect(data.url).to.be.a('string');
                expect(data.url).to.be("");
              }
            }
            expect(data.previous).to.be.a('string');
            expect(data.previous).to.be("");
            expect(data.next).to.be.a('string');
            expect(data.next).to.be("");

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
      describe('listGitAccesses', function() {
        it('should call listGitAccesses successfully', function(done) {
          // TODO: uncomment, update parameter values for listGitAccesses call and complete the assertions
          /*
          var owner = "owner_example";
          var opts = {};
          opts.offset = 56;
          opts.limit = 56;
          opts.sort = "sort_example";
          opts.query = "query_example";

          instance.listGitAccesses(owner, opts, function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1ListHostAccessesResponse);
            expect(data.count).to.be.a('number');
            expect(data.count).to.be(0);
            {
              let dataCtr = data.results;
              expect(dataCtr).to.be.an(Array);
              expect(dataCtr).to.not.be.empty();
              for (let p in dataCtr) {
                let data = dataCtr[p];
                expect(data).to.be.a(PolyaxonSdk.V1HostAccess);
                expect(data.uuid).to.be.a('string');
                expect(data.uuid).to.be("");
                expect(data.name).to.be.a('string');
                expect(data.name).to.be("");
                expect(data.description).to.be.a('string');
                expect(data.description).to.be("");
                expect(data.readme).to.be.a('string');
                expect(data.readme).to.be("");
                {
                  let dataCtr = data.tags;
                  expect(dataCtr).to.be.an(Array);
                  expect(dataCtr).to.not.be.empty();
                  for (let p in dataCtr) {
                    let data = dataCtr[p];
                    expect(data).to.be.a('string');
                    expect(data).to.be("");
                  }
                }
                expect(data.created_at).to.be.a(Date);
                expect(data.created_at).to.be(new Date());
                expect(data.updated_at).to.be.a(Date);
                expect(data.updated_at).to.be(new Date());
                expect(data.frozen).to.be.a('boolean');
                expect(data.frozen).to.be(false);
                expect(data.disabled).to.be.a('boolean');
                expect(data.disabled).to.be(false);
                expect(data.insecure).to.be.a('boolean');
                expect(data.insecure).to.be(false);
                expect(data.deleted).to.be.a('boolean');
                expect(data.deleted).to.be(false);
                expect(data.k8s_secret).to.be.a('string');
                expect(data.k8s_secret).to.be("");
                expect(data.url).to.be.a('string');
                expect(data.url).to.be("");
              }
            }
            expect(data.previous).to.be.a('string');
            expect(data.previous).to.be("");
            expect(data.next).to.be.a('string');
            expect(data.next).to.be("");

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
      describe('patchGitAccess', function() {
        it('should call patchGitAccess successfully', function(done) {
          // TODO: uncomment, update parameter values for patchGitAccess call and complete the assertions
          /*
          var owner = "owner_example";
          var host_access_uuid = "host_access_uuid_example";
          var body = new PolyaxonSdk.V1HostAccess();
          body.uuid = "";
          body.name = "";
          body.description = "";
          body.readme = "";
          body.tags = [""];
          body.created_at = new Date();
          body.updated_at = new Date();
          body.frozen = false;
          body.disabled = false;
          body.insecure = false;
          body.deleted = false;
          body.k8s_secret = "";
          body.url = "";

          instance.patchGitAccess(owner, host_access_uuid, body, function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1HostAccess);
            expect(data.uuid).to.be.a('string');
            expect(data.uuid).to.be("");
            expect(data.name).to.be.a('string');
            expect(data.name).to.be("");
            expect(data.description).to.be.a('string');
            expect(data.description).to.be("");
            expect(data.readme).to.be.a('string');
            expect(data.readme).to.be("");
            {
              let dataCtr = data.tags;
              expect(dataCtr).to.be.an(Array);
              expect(dataCtr).to.not.be.empty();
              for (let p in dataCtr) {
                let data = dataCtr[p];
                expect(data).to.be.a('string');
                expect(data).to.be("");
              }
            }
            expect(data.created_at).to.be.a(Date);
            expect(data.created_at).to.be(new Date());
            expect(data.updated_at).to.be.a(Date);
            expect(data.updated_at).to.be(new Date());
            expect(data.frozen).to.be.a('boolean');
            expect(data.frozen).to.be(false);
            expect(data.disabled).to.be.a('boolean');
            expect(data.disabled).to.be(false);
            expect(data.insecure).to.be.a('boolean');
            expect(data.insecure).to.be(false);
            expect(data.deleted).to.be.a('boolean');
            expect(data.deleted).to.be(false);
            expect(data.k8s_secret).to.be.a('string');
            expect(data.k8s_secret).to.be("");
            expect(data.url).to.be.a('string');
            expect(data.url).to.be("");

            done();
          });
          */
          // TODO: uncomment and complete method invocation above, then delete this line and the next:
          done();
        });
      });
      describe('updateGitAccess', function() {
        it('should call updateGitAccess successfully', function(done) {
          // TODO: uncomment, update parameter values for updateGitAccess call and complete the assertions
          /*
          var owner = "owner_example";
          var host_access_uuid = "host_access_uuid_example";
          var body = new PolyaxonSdk.V1HostAccess();
          body.uuid = "";
          body.name = "";
          body.description = "";
          body.readme = "";
          body.tags = [""];
          body.created_at = new Date();
          body.updated_at = new Date();
          body.frozen = false;
          body.disabled = false;
          body.insecure = false;
          body.deleted = false;
          body.k8s_secret = "";
          body.url = "";

          instance.updateGitAccess(owner, host_access_uuid, body, function(error, data, response) {
            if (error) {
              done(error);
              return;
            }
            // TODO: update response assertions
            expect(data).to.be.a(PolyaxonSdk.V1HostAccess);
            expect(data.uuid).to.be.a('string');
            expect(data.uuid).to.be("");
            expect(data.name).to.be.a('string');
            expect(data.name).to.be("");
            expect(data.description).to.be.a('string');
            expect(data.description).to.be("");
            expect(data.readme).to.be.a('string');
            expect(data.readme).to.be("");
            {
              let dataCtr = data.tags;
              expect(dataCtr).to.be.an(Array);
              expect(dataCtr).to.not.be.empty();
              for (let p in dataCtr) {
                let data = dataCtr[p];
                expect(data).to.be.a('string');
                expect(data).to.be("");
              }
            }
            expect(data.created_at).to.be.a(Date);
            expect(data.created_at).to.be(new Date());
            expect(data.updated_at).to.be.a(Date);
            expect(data.updated_at).to.be(new Date());
            expect(data.frozen).to.be.a('boolean');
            expect(data.frozen).to.be(false);
            expect(data.disabled).to.be.a('boolean');
            expect(data.disabled).to.be(false);
            expect(data.insecure).to.be.a('boolean');
            expect(data.insecure).to.be(false);
            expect(data.deleted).to.be.a('boolean');
            expect(data.deleted).to.be(false);
            expect(data.k8s_secret).to.be.a('string');
            expect(data.k8s_secret).to.be("");
            expect(data.url).to.be.a('string');
            expect(data.url).to.be("");

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
