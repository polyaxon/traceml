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

// Code generated by go-swagger; DO NOT EDIT.

package runs_v1

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	"fmt"
	"io"

	"github.com/go-openapi/runtime"
	"github.com/go-openapi/strfmt"

	"github.com/polyaxon/polyaxon/sdks/go/http_client/v1/service_model"
)

// DeleteRunArtifactReader is a Reader for the DeleteRunArtifact structure.
type DeleteRunArtifactReader struct {
	formats strfmt.Registry
}

// ReadResponse reads a server response into the received o.
func (o *DeleteRunArtifactReader) ReadResponse(response runtime.ClientResponse, consumer runtime.Consumer) (interface{}, error) {
	switch response.Code() {
	case 200:
		result := NewDeleteRunArtifactOK()
		if err := result.readResponse(response, consumer, o.formats); err != nil {
			return nil, err
		}
		return result, nil
	case 204:
		result := NewDeleteRunArtifactNoContent()
		if err := result.readResponse(response, consumer, o.formats); err != nil {
			return nil, err
		}
		return result, nil
	case 403:
		result := NewDeleteRunArtifactForbidden()
		if err := result.readResponse(response, consumer, o.formats); err != nil {
			return nil, err
		}
		return nil, result
	case 404:
		result := NewDeleteRunArtifactNotFound()
		if err := result.readResponse(response, consumer, o.formats); err != nil {
			return nil, err
		}
		return nil, result
	default:
		result := NewDeleteRunArtifactDefault(response.Code())
		if err := result.readResponse(response, consumer, o.formats); err != nil {
			return nil, err
		}
		if response.Code()/100 == 2 {
			return result, nil
		}
		return nil, result
	}
}

// NewDeleteRunArtifactOK creates a DeleteRunArtifactOK with default headers values
func NewDeleteRunArtifactOK() *DeleteRunArtifactOK {
	return &DeleteRunArtifactOK{}
}

/*DeleteRunArtifactOK handles this case with default header values.

A successful response.
*/
type DeleteRunArtifactOK struct {
	Payload interface{}
}

func (o *DeleteRunArtifactOK) Error() string {
	return fmt.Sprintf("[DELETE /streams/v1/{namespace}/{owner}/{project}/runs/{uuid}/artifact][%d] deleteRunArtifactOK  %+v", 200, o.Payload)
}

func (o *DeleteRunArtifactOK) GetPayload() interface{} {
	return o.Payload
}

func (o *DeleteRunArtifactOK) readResponse(response runtime.ClientResponse, consumer runtime.Consumer, formats strfmt.Registry) error {

	// response payload
	if err := consumer.Consume(response.Body(), &o.Payload); err != nil && err != io.EOF {
		return err
	}

	return nil
}

// NewDeleteRunArtifactNoContent creates a DeleteRunArtifactNoContent with default headers values
func NewDeleteRunArtifactNoContent() *DeleteRunArtifactNoContent {
	return &DeleteRunArtifactNoContent{}
}

/*DeleteRunArtifactNoContent handles this case with default header values.

No content.
*/
type DeleteRunArtifactNoContent struct {
	Payload interface{}
}

func (o *DeleteRunArtifactNoContent) Error() string {
	return fmt.Sprintf("[DELETE /streams/v1/{namespace}/{owner}/{project}/runs/{uuid}/artifact][%d] deleteRunArtifactNoContent  %+v", 204, o.Payload)
}

func (o *DeleteRunArtifactNoContent) GetPayload() interface{} {
	return o.Payload
}

func (o *DeleteRunArtifactNoContent) readResponse(response runtime.ClientResponse, consumer runtime.Consumer, formats strfmt.Registry) error {

	// response payload
	if err := consumer.Consume(response.Body(), &o.Payload); err != nil && err != io.EOF {
		return err
	}

	return nil
}

// NewDeleteRunArtifactForbidden creates a DeleteRunArtifactForbidden with default headers values
func NewDeleteRunArtifactForbidden() *DeleteRunArtifactForbidden {
	return &DeleteRunArtifactForbidden{}
}

/*DeleteRunArtifactForbidden handles this case with default header values.

You don't have permission to access the resource.
*/
type DeleteRunArtifactForbidden struct {
	Payload interface{}
}

func (o *DeleteRunArtifactForbidden) Error() string {
	return fmt.Sprintf("[DELETE /streams/v1/{namespace}/{owner}/{project}/runs/{uuid}/artifact][%d] deleteRunArtifactForbidden  %+v", 403, o.Payload)
}

func (o *DeleteRunArtifactForbidden) GetPayload() interface{} {
	return o.Payload
}

func (o *DeleteRunArtifactForbidden) readResponse(response runtime.ClientResponse, consumer runtime.Consumer, formats strfmt.Registry) error {

	// response payload
	if err := consumer.Consume(response.Body(), &o.Payload); err != nil && err != io.EOF {
		return err
	}

	return nil
}

// NewDeleteRunArtifactNotFound creates a DeleteRunArtifactNotFound with default headers values
func NewDeleteRunArtifactNotFound() *DeleteRunArtifactNotFound {
	return &DeleteRunArtifactNotFound{}
}

/*DeleteRunArtifactNotFound handles this case with default header values.

Resource does not exist.
*/
type DeleteRunArtifactNotFound struct {
	Payload interface{}
}

func (o *DeleteRunArtifactNotFound) Error() string {
	return fmt.Sprintf("[DELETE /streams/v1/{namespace}/{owner}/{project}/runs/{uuid}/artifact][%d] deleteRunArtifactNotFound  %+v", 404, o.Payload)
}

func (o *DeleteRunArtifactNotFound) GetPayload() interface{} {
	return o.Payload
}

func (o *DeleteRunArtifactNotFound) readResponse(response runtime.ClientResponse, consumer runtime.Consumer, formats strfmt.Registry) error {

	// response payload
	if err := consumer.Consume(response.Body(), &o.Payload); err != nil && err != io.EOF {
		return err
	}

	return nil
}

// NewDeleteRunArtifactDefault creates a DeleteRunArtifactDefault with default headers values
func NewDeleteRunArtifactDefault(code int) *DeleteRunArtifactDefault {
	return &DeleteRunArtifactDefault{
		_statusCode: code,
	}
}

/*DeleteRunArtifactDefault handles this case with default header values.

An unexpected error response
*/
type DeleteRunArtifactDefault struct {
	_statusCode int

	Payload *service_model.RuntimeError
}

// Code gets the status code for the delete run artifact default response
func (o *DeleteRunArtifactDefault) Code() int {
	return o._statusCode
}

func (o *DeleteRunArtifactDefault) Error() string {
	return fmt.Sprintf("[DELETE /streams/v1/{namespace}/{owner}/{project}/runs/{uuid}/artifact][%d] DeleteRunArtifact default  %+v", o._statusCode, o.Payload)
}

func (o *DeleteRunArtifactDefault) GetPayload() *service_model.RuntimeError {
	return o.Payload
}

func (o *DeleteRunArtifactDefault) readResponse(response runtime.ClientResponse, consumer runtime.Consumer, formats strfmt.Registry) error {

	o.Payload = new(service_model.RuntimeError)

	// response payload
	if err := consumer.Consume(response.Body(), o.Payload); err != nil && err != io.EOF {
		return err
	}

	return nil
}
