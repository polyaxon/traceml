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
 * The version of the OpenAPI document: 1.2.0-rc3
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


import * as runtime from '../runtime';
import {
    RuntimeError,
    RuntimeErrorFromJSON,
    RuntimeErrorToJSON,
    V1ListModelRegistryResponse,
    V1ListModelRegistryResponseFromJSON,
    V1ListModelRegistryResponseToJSON,
    V1ModelRegistry,
    V1ModelRegistryFromJSON,
    V1ModelRegistryToJSON,
} from '../models';

export interface CreateModelRegistryRequest {
    owner: string;
    body: V1ModelRegistry;
}

export interface DeleteModelRegistryRequest {
    owner: string;
    uuid: string;
}

export interface GetModelRegistryRequest {
    owner: string;
    uuid: string;
}

export interface ListModelRegistryRequest {
    owner: string;
    offset?: number;
    limit?: number;
    sort?: string;
    query?: string;
}

export interface ListModelRegistryNamesRequest {
    owner: string;
    offset?: number;
    limit?: number;
    sort?: string;
    query?: string;
}

export interface PatchModelRegistryRequest {
    owner: string;
    modelUuid: string;
    body: V1ModelRegistry;
}

export interface UpdateModelRegistryRequest {
    owner: string;
    modelUuid: string;
    body: V1ModelRegistry;
}

/**
 * 
 */
export class ModelRegistryV1Api extends runtime.BaseAPI {

    /**
     * Create hub model
     */
    async createModelRegistryRaw(requestParameters: CreateModelRegistryRequest): Promise<runtime.ApiResponse<V1ModelRegistry>> {
        if (requestParameters.owner === null || requestParameters.owner === undefined) {
            throw new runtime.RequiredError('owner','Required parameter requestParameters.owner was null or undefined when calling createModelRegistry.');
        }

        if (requestParameters.body === null || requestParameters.body === undefined) {
            throw new runtime.RequiredError('body','Required parameter requestParameters.body was null or undefined when calling createModelRegistry.');
        }

        const queryParameters: runtime.HTTPQuery = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["Authorization"] = this.configuration.apiKey("Authorization"); // ApiKey authentication
        }

        const response = await this.request({
            path: `/api/v1/orgs/{owner}/models`.replace(`{${"owner"}}`, encodeURIComponent(String(requestParameters.owner))),
            method: 'POST',
            headers: headerParameters,
            query: queryParameters,
            body: V1ModelRegistryToJSON(requestParameters.body),
        });

        return new runtime.JSONApiResponse(response, (jsonValue) => V1ModelRegistryFromJSON(jsonValue));
    }

    /**
     * Create hub model
     */
    async createModelRegistry(requestParameters: CreateModelRegistryRequest): Promise<V1ModelRegistry> {
        const response = await this.createModelRegistryRaw(requestParameters);
        return await response.value();
    }

    /**
     * Delete hub model
     */
    async deleteModelRegistryRaw(requestParameters: DeleteModelRegistryRequest): Promise<runtime.ApiResponse<object>> {
        if (requestParameters.owner === null || requestParameters.owner === undefined) {
            throw new runtime.RequiredError('owner','Required parameter requestParameters.owner was null or undefined when calling deleteModelRegistry.');
        }

        if (requestParameters.uuid === null || requestParameters.uuid === undefined) {
            throw new runtime.RequiredError('uuid','Required parameter requestParameters.uuid was null or undefined when calling deleteModelRegistry.');
        }

        const queryParameters: runtime.HTTPQuery = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["Authorization"] = this.configuration.apiKey("Authorization"); // ApiKey authentication
        }

        const response = await this.request({
            path: `/api/v1/orgs/{owner}/models/{uuid}`.replace(`{${"owner"}}`, encodeURIComponent(String(requestParameters.owner))).replace(`{${"uuid"}}`, encodeURIComponent(String(requestParameters.uuid))),
            method: 'DELETE',
            headers: headerParameters,
            query: queryParameters,
        });

        return new runtime.JSONApiResponse<any>(response);
    }

    /**
     * Delete hub model
     */
    async deleteModelRegistry(requestParameters: DeleteModelRegistryRequest): Promise<object> {
        const response = await this.deleteModelRegistryRaw(requestParameters);
        return await response.value();
    }

    /**
     * Get hub model
     */
    async getModelRegistryRaw(requestParameters: GetModelRegistryRequest): Promise<runtime.ApiResponse<V1ModelRegistry>> {
        if (requestParameters.owner === null || requestParameters.owner === undefined) {
            throw new runtime.RequiredError('owner','Required parameter requestParameters.owner was null or undefined when calling getModelRegistry.');
        }

        if (requestParameters.uuid === null || requestParameters.uuid === undefined) {
            throw new runtime.RequiredError('uuid','Required parameter requestParameters.uuid was null or undefined when calling getModelRegistry.');
        }

        const queryParameters: runtime.HTTPQuery = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["Authorization"] = this.configuration.apiKey("Authorization"); // ApiKey authentication
        }

        const response = await this.request({
            path: `/api/v1/orgs/{owner}/models/{uuid}`.replace(`{${"owner"}}`, encodeURIComponent(String(requestParameters.owner))).replace(`{${"uuid"}}`, encodeURIComponent(String(requestParameters.uuid))),
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        });

        return new runtime.JSONApiResponse(response, (jsonValue) => V1ModelRegistryFromJSON(jsonValue));
    }

    /**
     * Get hub model
     */
    async getModelRegistry(requestParameters: GetModelRegistryRequest): Promise<V1ModelRegistry> {
        const response = await this.getModelRegistryRaw(requestParameters);
        return await response.value();
    }

    /**
     * List hub models
     */
    async listModelRegistryRaw(requestParameters: ListModelRegistryRequest): Promise<runtime.ApiResponse<V1ListModelRegistryResponse>> {
        if (requestParameters.owner === null || requestParameters.owner === undefined) {
            throw new runtime.RequiredError('owner','Required parameter requestParameters.owner was null or undefined when calling listModelRegistry.');
        }

        const queryParameters: runtime.HTTPQuery = {};

        if (requestParameters.offset !== undefined) {
            queryParameters['offset'] = requestParameters.offset;
        }

        if (requestParameters.limit !== undefined) {
            queryParameters['limit'] = requestParameters.limit;
        }

        if (requestParameters.sort !== undefined) {
            queryParameters['sort'] = requestParameters.sort;
        }

        if (requestParameters.query !== undefined) {
            queryParameters['query'] = requestParameters.query;
        }

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["Authorization"] = this.configuration.apiKey("Authorization"); // ApiKey authentication
        }

        const response = await this.request({
            path: `/api/v1/orgs/{owner}/models`.replace(`{${"owner"}}`, encodeURIComponent(String(requestParameters.owner))),
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        });

        return new runtime.JSONApiResponse(response, (jsonValue) => V1ListModelRegistryResponseFromJSON(jsonValue));
    }

    /**
     * List hub models
     */
    async listModelRegistry(requestParameters: ListModelRegistryRequest): Promise<V1ListModelRegistryResponse> {
        const response = await this.listModelRegistryRaw(requestParameters);
        return await response.value();
    }

    /**
     * List hub model names
     */
    async listModelRegistryNamesRaw(requestParameters: ListModelRegistryNamesRequest): Promise<runtime.ApiResponse<V1ListModelRegistryResponse>> {
        if (requestParameters.owner === null || requestParameters.owner === undefined) {
            throw new runtime.RequiredError('owner','Required parameter requestParameters.owner was null or undefined when calling listModelRegistryNames.');
        }

        const queryParameters: runtime.HTTPQuery = {};

        if (requestParameters.offset !== undefined) {
            queryParameters['offset'] = requestParameters.offset;
        }

        if (requestParameters.limit !== undefined) {
            queryParameters['limit'] = requestParameters.limit;
        }

        if (requestParameters.sort !== undefined) {
            queryParameters['sort'] = requestParameters.sort;
        }

        if (requestParameters.query !== undefined) {
            queryParameters['query'] = requestParameters.query;
        }

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["Authorization"] = this.configuration.apiKey("Authorization"); // ApiKey authentication
        }

        const response = await this.request({
            path: `/api/v1/orgs/{owner}/models/names`.replace(`{${"owner"}}`, encodeURIComponent(String(requestParameters.owner))),
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        });

        return new runtime.JSONApiResponse(response, (jsonValue) => V1ListModelRegistryResponseFromJSON(jsonValue));
    }

    /**
     * List hub model names
     */
    async listModelRegistryNames(requestParameters: ListModelRegistryNamesRequest): Promise<V1ListModelRegistryResponse> {
        const response = await this.listModelRegistryNamesRaw(requestParameters);
        return await response.value();
    }

    /**
     * Patch hub model
     */
    async patchModelRegistryRaw(requestParameters: PatchModelRegistryRequest): Promise<runtime.ApiResponse<V1ModelRegistry>> {
        if (requestParameters.owner === null || requestParameters.owner === undefined) {
            throw new runtime.RequiredError('owner','Required parameter requestParameters.owner was null or undefined when calling patchModelRegistry.');
        }

        if (requestParameters.modelUuid === null || requestParameters.modelUuid === undefined) {
            throw new runtime.RequiredError('modelUuid','Required parameter requestParameters.modelUuid was null or undefined when calling patchModelRegistry.');
        }

        if (requestParameters.body === null || requestParameters.body === undefined) {
            throw new runtime.RequiredError('body','Required parameter requestParameters.body was null or undefined when calling patchModelRegistry.');
        }

        const queryParameters: runtime.HTTPQuery = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["Authorization"] = this.configuration.apiKey("Authorization"); // ApiKey authentication
        }

        const response = await this.request({
            path: `/api/v1/orgs/{owner}/models/{model.uuid}`.replace(`{${"owner"}}`, encodeURIComponent(String(requestParameters.owner))).replace(`{${"model.uuid"}}`, encodeURIComponent(String(requestParameters.modelUuid))),
            method: 'PATCH',
            headers: headerParameters,
            query: queryParameters,
            body: V1ModelRegistryToJSON(requestParameters.body),
        });

        return new runtime.JSONApiResponse(response, (jsonValue) => V1ModelRegistryFromJSON(jsonValue));
    }

    /**
     * Patch hub model
     */
    async patchModelRegistry(requestParameters: PatchModelRegistryRequest): Promise<V1ModelRegistry> {
        const response = await this.patchModelRegistryRaw(requestParameters);
        return await response.value();
    }

    /**
     * Update hub model
     */
    async updateModelRegistryRaw(requestParameters: UpdateModelRegistryRequest): Promise<runtime.ApiResponse<V1ModelRegistry>> {
        if (requestParameters.owner === null || requestParameters.owner === undefined) {
            throw new runtime.RequiredError('owner','Required parameter requestParameters.owner was null or undefined when calling updateModelRegistry.');
        }

        if (requestParameters.modelUuid === null || requestParameters.modelUuid === undefined) {
            throw new runtime.RequiredError('modelUuid','Required parameter requestParameters.modelUuid was null or undefined when calling updateModelRegistry.');
        }

        if (requestParameters.body === null || requestParameters.body === undefined) {
            throw new runtime.RequiredError('body','Required parameter requestParameters.body was null or undefined when calling updateModelRegistry.');
        }

        const queryParameters: runtime.HTTPQuery = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["Authorization"] = this.configuration.apiKey("Authorization"); // ApiKey authentication
        }

        const response = await this.request({
            path: `/api/v1/orgs/{owner}/models/{model.uuid}`.replace(`{${"owner"}}`, encodeURIComponent(String(requestParameters.owner))).replace(`{${"model.uuid"}}`, encodeURIComponent(String(requestParameters.modelUuid))),
            method: 'PUT',
            headers: headerParameters,
            query: queryParameters,
            body: V1ModelRegistryToJSON(requestParameters.body),
        });

        return new runtime.JSONApiResponse(response, (jsonValue) => V1ModelRegistryFromJSON(jsonValue));
    }

    /**
     * Update hub model
     */
    async updateModelRegistry(requestParameters: UpdateModelRegistryRequest): Promise<V1ModelRegistry> {
        const response = await this.updateModelRegistryRaw(requestParameters);
        return await response.value();
    }

}
