import { defHttp } from '/@/utils/http/axios';

/**
 * Assign data to a user
 */
export function assignDataApi(params: {
  datasetId: number;
  userId: number;
  dataIds: number[];
}) {
  return defHttp.post({
    url: '/api/v1/assignments',
    data: params
  });
}

/**
 * Get data assignments
 */
export function getAssignmentsApi(params: {
  datasetId?: number;
  userId?: number;
  status?: string;
  pageNo?: number;
  pageSize?: number;
}) {
  return defHttp.get({
    url: '/api/v1/assignments',
    params
  });
}

/**
 * Remove data assignment
 */
export function removeAssignmentApi(assignmentId: number) {
  return defHttp.delete({
    url: `/api/v1/assignments/${assignmentId}`
  });
}

/**
 * Check data access
 */
export function checkDataAccessApi(dataId: number, userId: number) {
  return defHttp.get({
    url: `/api/v1/assignments/access/${dataId}`,
    params: { userId }
  });
} 